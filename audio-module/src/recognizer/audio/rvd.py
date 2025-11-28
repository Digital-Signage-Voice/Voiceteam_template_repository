import os
import time
import traceback
import numpy as np
import librosa
import noisereduce as nr
import scipy.ndimage as nd
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

# -----------------------------
# 설정 관리 (Configuration)
# -----------------------------
@dataclass
class NoiseReducerConfig:
    """잡음 제거 모듈의 하이퍼파라미터를 관리하는 설정 클래스"""
    # 오디오 기본 설정
    sample_rate: int = 16000
    n_fft: int = 2048
    hop_length: int = 512
    
    # 노이즈 프로파일링 관련
    initial_noise_db: float = -40.0
    profile_init_duration_sec: float = 3.0
    min_buffer_frames_for_update: int = 4
    relearn_trigger_ratio: float = 0.30
    
    # 동기화 및 지연(Latency) 관련
    sync_tolerance_sec: float = 0.04    # 40ms 허용 오차
    max_wait_for_vad_sec: float = 0.03  # VAD 대기 최대 30ms
    low_latency_mode: bool = True       # 저지연 모드 우선
    
    # 알고리즘 파라미터
    stationary: bool = False
    stage1_prop_decrease: float = 0.98
    smoothing_kernel_size: int = 3
    vad_confidence_threshold: float = 0.8
    non_speech_gain: float = 0.05
    
    # 디버깅/안전장치
    bypass_mode: bool = False  # True일 경우 노이즈 제거 없이 통과 (비상용)


class FinalNoiseReducer:
    """
    [Real-time Audio Noise Reducer]
    비동기 VAD 신호와 오디오 스트림을 큐(Queue)로 동기화하여 
    저지연으로 잡음을 제거하는 모듈입니다.
    """
    
    def __init__(self, config: NoiseReducerConfig = NoiseReducerConfig()):
        self.cfg = config
        
        # 상태 변수 초기화
        self._init_state_variables()
        self._init_buffers()
        
        # 노이즈 프로파일 초기화
        self._init_noise_profile()

        print(f"[NoiseReducer] Ready. SR={self.cfg.sample_rate}, LatencyMode={self.cfg.low_latency_mode}")

    def _init_state_variables(self):
        """내부 상태 변수 초기화"""
        self.profile_state = "initializing"
        self.frames_seen = 0
        self.last_noise_mean = None
        self.highest_seen_ts = 0.0
        
        # Fallback용 마지막 VAD 상태 (기본값: 미발화)
        self.last_vad_state = (False, 0.0) 
        
        # Overlap-Add 처리를 위한 이전 청크 꼬리 부분
        self.overlap_len = max(0, self.cfg.n_fft - self.cfg.hop_length)
        self.prev_input_tail = np.zeros(self.overlap_len, dtype=np.float32)

    def _init_buffers(self):
        """데이터 버퍼링을 위한 큐 초기화"""
        self.audio_buffer = deque()  # (timestamp, audio_chunk)
        self.vad_buffer = deque()    # (timestamp, is_speaking, confidence)
        self.output_queue = deque()  # processed_audio_chunk
        self.recent_magnitudes = []  # 프로파일 업데이트용 버퍼

    def _init_noise_profile(self):
        """초기 노이즈 프로파일 생성"""
        num_bins = self.cfg.n_fft // 2 + 1
        init_power = 10 ** (self.cfg.initial_noise_db / 10.0)
        init_mag = np.sqrt(init_power)
        self.noise_profile = np.full((num_bins, 1), init_mag, dtype=np.float32)
        
        # 프로파일링에 필요한 프레임 수 계산
        self.initial_frames_needed = max(1, int(self.cfg.profile_init_duration_sec * self.cfg.sample_rate / self.cfg.hop_length))

    # -------------------------
    # Public API
    # -------------------------
    def add_audio_chunk(self, audio_chunk: np.ndarray, timestamp_sec: float):
        """오디오 청크 입력 (타임스탬프 필수)"""
        self.audio_buffer.append((timestamp_sec, audio_chunk))
        self.highest_seen_ts = max(self.highest_seen_ts, timestamp_sec)
        self._process_buffers() # 데이터 들어오면 즉시 처리 시도

    def add_vad_result(self, is_speaking: bool, vad_confidence: float, timestamp_sec: float):
        """VAD 결과 입력 (비동기)"""
        self.vad_buffer.append((timestamp_sec, is_speaking, float(vad_confidence)))
        
        # 타임스탬프 순서 보장 (네트워크 지터 대비)
        if len(self.vad_buffer) > 1 and self.vad_buffer[-1][0] < self.vad_buffer[-2][0]:
            self.vad_buffer = deque(sorted(list(self.vad_buffer), key=lambda x: x[0]))
            
        self.highest_seen_ts = max(self.highest_seen_ts, timestamp_sec)
        self._process_buffers()

    def get_processed_chunk(self) -> Optional[np.ndarray]:
        """처리된 청크 반환 (없으면 None)"""
        if self.output_queue:
            return self.output_queue.popleft()
        return None
    
    def set_bypass(self, enabled: bool):
        """[Emergency] 노이즈 제거 기능을 끄고 패스스루 모드로 전환"""
        self.cfg.bypass_mode = enabled
        print(f"[NoiseReducer] Bypass mode: {enabled}")

    def flush(self) -> np.ndarray:
        """남은 버퍼 강제 처리 및 반환"""
        print("[NoiseReducer] Flushing buffers...")
        flushed_chunks = []
        
        while self.audio_buffer:
            ts, chunk = self.audio_buffer.popleft()
            # 마지막 알려진 VAD 상태로 처리
            is_speaking, conf = self.last_vad_state
            processed = self._denoise_core(chunk, is_speaking, conf)
            flushed_chunks.append(processed)
            
        while self.output_queue:
            flushed_chunks.append(self.output_queue.popleft())
            
        return np.concatenate(flushed_chunks) if flushed_chunks else np.array([], dtype=np.float32)

    # -------------------------
    # Core Logic (Synchronization)
    # -------------------------
    def _process_buffers(self):
        """
        오디오와 VAD의 타임스탬프를 매칭(Sync)하여 처리하는 핵심 로직.
        전략: 매칭 성공 -> 즉시 처리 / 매칭 실패 -> 대기 or 타임아웃(Fallback)
        """
        while self.audio_buffer:
            audio_ts, audio_chunk = self.audio_buffer[0]
            
            # 1. Bypass 모드면 즉시 통과
            if self.cfg.bypass_mode:
                self.audio_buffer.popleft()
                self.output_queue.append(audio_chunk)
                continue

            # VAD 정보가 없으면 대기 (단, 타임아웃 체크)
            if not self.vad_buffer:
                waited = self.highest_seen_ts - audio_ts
                if waited >= self.cfg.max_wait_for_vad_sec or self.cfg.low_latency_mode:
                     # 타임아웃 or 저지연 모드 -> 마지막 상태로 Fallback 처리
                    self._fallback_processing()
                else:
                    break # VAD 기다림
                continue

            # VAD 정보가 있음 -> 매칭 시도
            vad_ts, is_speaking, vad_conf = self.vad_buffer[0]
            diff = audio_ts - vad_ts

            if abs(diff) <= self.cfg.sync_tolerance_sec:
                # [Case 1] 매칭 성공
                self.audio_buffer.popleft()
                self.vad_buffer.popleft()
                self.last_vad_state = (is_speaking, vad_conf)
                
                processed = self._denoise_core(audio_chunk, is_speaking, vad_conf)
                self.output_queue.append(processed)

            elif diff > self.cfg.sync_tolerance_sec:
                # [Case 2] VAD가 너무 과거의 것 (Stale) -> 폐기하고 다음 VAD 확인
                self.vad_buffer.popleft()

            else: # diff < -tolerance (오디오가 VAD보다 훨씬 과거)
                # [Case 3] 해당 오디오에 맞는 VAD가 아직 안 옴
                waited = self.highest_seen_ts - audio_ts
                if waited >= self.cfg.max_wait_for_vad_sec:
                    self._fallback_processing() # 너무 오래 기다림 -> Fallback
                else:
                    break # 더 기다림

    def _fallback_processing(self):
        """VAD 없이 마지막 상태로 강제 처리 (Latency 방어)"""
        if not self.audio_buffer: return
        _, audio_chunk = self.audio_buffer.popleft()
        is_speaking, conf = self.last_vad_state
        processed = self._denoise_core(audio_chunk, is_speaking, conf)
        self.output_queue.append(processed)

    # -------------------------
    # DSP Logic (Denoising)
    # -------------------------
    def _denoise_core(self, audio_chunk: np.ndarray, is_speaking: bool, vad_conf: float) -> np.ndarray:
        """실제 노이즈 제거 알고리즘 수행"""
        if len(audio_chunk) == 0: return audio_chunk
        
        # 0. 비발화 구간이면 강제 감쇠 (Gain 조절)
        if not is_speaking:
            # 프로파일 업데이트 (비발화 구간 학습)
            self._update_profile_if_needed(audio_chunk)
            return audio_chunk * self.cfg.non_speech_gain

        try:
            # 1. Noise Reduction (Spectral Gating)
            # noisereduce 라이브러리가 무거우면 여기서 try-except로 bypass 하도록 수정
            denoised = nr.reduce_noise(
                y=audio_chunk,
                sr=self.cfg.sample_rate,
                y_noise=self.noise_profile,
                prop_decrease=self.cfg.stage1_prop_decrease,
                n_fft=self.cfg.n_fft,
                hop_length=self.cfg.hop_length,
                stationary=self.cfg.stationary
            )
            
            # 2. Smoothing (VAD 신뢰도 기반)
            # 신뢰도가 낮으면 원본 소리를 좀 더 섞어서 왜곡 방지
            if self.cfg.smoothing_kernel_size > 0:
                 denoised = self._apply_smoothing(denoised, vad_conf)
            
            return denoised

        except Exception as e:
            print(f"[ERROR] Denoising fail: {e}")
            # 에러 발생 시 원본 반환 (소리가 끊기는 것보다 잡음 있는 게 나음)
            return audio_chunk

    def _update_profile_if_needed(self, audio_chunk):
        """비발화 구간의 오디오로 노이즈 프로파일 업데이트"""
        try:
            # STFT 계산
            stft = librosa.stft(audio_chunk, n_fft=self.cfg.n_fft, hop_length=self.cfg.hop_length, center=False)
            mag = np.abs(stft)
            
            # 차원 맞추기 (가끔 n_fft 패딩 문제 발생 방지)
            if mag.shape[0] != self.noise_profile.shape[0]:
                return 

            self.recent_magnitudes.append(mag)
            if len(self.recent_magnitudes) > self.cfg.min_buffer_frames_for_update:
                self.recent_magnitudes.pop(0)
                
                # 이동 평균으로 프로파일 갱신
                buffered = np.hstack(self.recent_magnitudes)
                current_noise = np.mean(buffered, axis=1, keepdims=True)
                
                update_factor = 0.1 # 학습 속도
                self.noise_profile = (1 - update_factor) * self.noise_profile + update_factor * current_noise
        except Exception:
            pass # 학습 실패는 치명적이지 않음

    def _apply_smoothing(self, audio, vad_conf):
        """VAD 신뢰도에 따른 스무딩 적용"""
        try:
            # 신뢰도가 높을수록 스무딩 적게 (선명하게)
            sigma = 1.0 * (1.0 - vad_conf) 
            if sigma < 0.1: return audio
            
            smoothed = nd.gaussian_filter1d(audio, sigma=sigma)
            return smoothed
        except:
            return audio

# ----------------------------------------------------------------------
# Usage Example (For Team Integration)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1. 설정 생성
    config = NoiseReducerConfig(
        sample_rate=16000,
        low_latency_mode=True,
        bypass_mode=False # 문제 생기면 True로 변경하세요
    )
    
    # 2. 인스턴스 생성
    reducer = FinalNoiseReducer(config)
    
    # 3. 데이터 주입 예시 (Pseudo Code)
    print("--- Hackathon Audio Pipeline Started ---")
    # while streaming:
    #     reducer.add_vad_result(is_speaking, conf, ts)
    #     reducer.add_audio_chunk(audio_data, ts)
    #     
    #     processed = reducer.get_processed_chunk()
    #     if processed is not None:
    #         send_to_stt(processed)