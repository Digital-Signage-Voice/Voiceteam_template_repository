# """
# FinalNoiseReducer (잡음 제거 모듈)

# 이 모듈은 실시간 오디오 스트림과 비동기 VAD(Voice Activity Detection)
# 정보를 입력받아, 저지연(Low-latency)으로 잡음을 제거하는 잡음 제거기입니다.

# 주요 클래스:
# - FinalNoiseReducer

# 주요 Public API:
# - add_audio_chunk(audio_chunk, timestamp_sec)
# - add_vad_result(is_speaking, vad_confidence, timestamp_sec)
# - get_processed_chunk()
# - flush()
# # STT 모듈과의 통합(호출) 방식 조언이 코드 맨 끝에 적혀 있습니다.
# # 필요하실 경우 참고하시면 됩니다.
# """

import os
import traceback
from collections import deque
import numpy as np
import librosa
import noisereduce as nr
import scipy.ndimage as nd

# --- 모듈 통합에 필요한 라이브러리 임포트 (테스트용) ---
import soundfile as sf # 오디오 파일 저장용

# -----------------------------
# 기본 상수 (모듈 공통)
# -----------------------------
DEFAULT_N_FFT = 2048
DEFAULT_HOP = 512
DEFAULT_SR = 16000 # STT 모듈과의 호환성을 위해 16kHz 권장


class FinalNoiseReducer:
    # """
    # FinalNoiseReducer
    # - 비동기 큐 기반 안정성 + 저지연(타임스탬프 엄격성) 아이디어
    # - Public API를 통해 오디오/VAD를 비동기(혹은 동기)로 입력받고,
    #   처리된 오디오 청크를 큐(Queue)를 통해 반환합니다.
    # """
    def __init__(self,
                 sample_rate=DEFAULT_SR,
                 n_fft=DEFAULT_N_FFT,
                 hop_length=DEFAULT_HOP,
                 stationary=False,
                 stage1_prop_decrease=0.98,
                 apply_smoothing=True,
                 smoothing_kernel_size=3,
                 save_stage1_output=True,
                 initial_noise_db=-40.0,
                 profile_init_duration_sec=3.0,
                 stable_update_factor=0.01,
                 init_update_factor=0.15,
                 relearn_trigger_ratio=0.30,
                 min_buffer_frames_for_update=4,
                 vad_confidence_threshold=0.8,
                 correlation_threshold=0.2,
                 non_speech_gain=0.05,

                 sync_tolerance_sec=0.04,     # 동기화 허용 오차 (40ms)
                 max_wait_for_vad_sec=0.03,   # VAD가 늦을 경우 최대 기다리는 시간 (30ms)
                 low_latency_mode=True        # True면 max_wait_for_vad_sec 만료 전이라도 처리
                 ):
        # 파라미터
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.stationary = stationary
        self.stage1_prop_decrease = float(stage1_prop_decrease)
        self.apply_smoothing = bool(apply_smoothing)
        self.smoothing_kernel_size = int(smoothing_kernel_size)
        self.save_stage1_output = bool(save_stage1_output)

        # 노이즈 프로파일 초기화
        num_bins = self.n_fft // 2 + 1
        init_power = 10 ** (initial_noise_db / 10.0)
        init_mag = np.sqrt(init_power)
        self.noise_profile = np.full((num_bins, 1), init_mag, dtype=np.float32)
        self.is_profile_ready = True

        # 최근 magnitude 버퍼 (프로파일링용)
        self.recent_magnitudes = []
        self.max_buffer_frames = max(1, int(profile_init_duration_sec * sample_rate / hop_length))

        # stage1 중간 저장 (디버깅용)
        self.stage1_results_buffer = [] if self.save_stage1_output else []

        # 프로파일 FSM (Finite State Machine)
        self.profile_state = "initializing"
        self.profile_init_duration_sec = float(profile_init_duration_sec)
        self.initial_frames_needed = max(1, int(self.profile_init_duration_sec * sample_rate / hop_length))
        self.frames_seen = 0
        self.last_noise_mean = None
        self.noise_floor_history = deque(maxlen=200)
        self.init_update_factor = float(init_update_factor)
        self.stable_update_factor = float(stable_update_factor)
        self.relearn_trigger_ratio = float(relearn_trigger_ratio)
        self.min_buffer_frames_for_update = max(1, int(min_buffer_frames_for_update))

        # 임계값
        self.vad_confidence_threshold = float(vad_confidence_threshold)
        self.correlation_threshold = float(correlation_threshold)
        self.non_speech_gain = float(non_speech_gain)

        # Overlap(중첩) 처리용 버퍼
        self.overlap_len = max(0, self.n_fft - self.hop_length)
        self.prev_input_tail = np.zeros(self.overlap_len, dtype=np.float32)

        # 하이브리드 동기화 버퍼 (입력/출력 큐)
        self.audio_buffer = deque()   # (타임스탬프_초, 오디오_청크)
        self.vad_buffer = deque()     # (타임스탬프_초, 발화여부, 신뢰도)
        self.output_queue = deque()   # (처리된_오디오_청크)

        # 매칭 및 지연 시간 튜닝
        self.sync_tolerance_sec = float(sync_tolerance_sec)
        self.max_wait_for_vad_sec = float(max_wait_for_vad_sec)
        self.low_latency_mode = bool(low_latency_mode)

        # 폴백(Fallback)을 위한 마지막으로 알려진 VAD 상태
        self.last_vad_state = (False, 0.0) # (is_speaking, vad_conf)

        # 대기 결정에 도움을 주는, 마지막으로 확인된 타임스탬프
        self.highest_seen_ts = 0.0

        print(f"[Reducer] 초기화 완료: sr={self.sample_rate}, n_fft={self.n_fft}, hop={self.hop_length}")
        print(f"[Reducer] 동기화 정책: 허용 오차={self.sync_tolerance_sec*1000:.1f}ms, 최대 대기={self.max_wait_for_vad_sec*1000:.1f}ms")

    # -------------------------
    # 공개 API (Public API)
    # -------------------------
    def add_audio_chunk(self, audio_chunk: np.ndarray, timestamp_sec: float):
        """
        [알림] 오디오 청크를 타임스탬프와 함께 버퍼에 추가합니다.
        """
        self.audio_buffer.append((timestamp_sec, audio_chunk))
        self.highest_seen_ts = max(self.highest_seen_ts, timestamp_sec)
        # 데이터가 추가되면 즉시 처리 시도
        self._try_process_buffers()

    def add_vad_result(self, is_speaking: bool, vad_confidence: float, timestamp_sec: float):
        """
        [알림] VAD 결과를 타임스탬프와 함께 버퍼에 추가합니다.
        """
        # 정렬 비용을 줄이기 위해 정렬 삽입 (간단히: 추가 후 순서 확인)
        self.vad_buffer.append((timestamp_sec, is_speaking, float(vad_confidence)))
        if len(self.vad_buffer) > 1 and self.vad_buffer[-1][0] < self.vad_buffer[-2][0]:
            # VAD가 순서대로 들어오지 않았을 경우 (네트워크 지터 등) 정렬
            self.vad_buffer = deque(sorted(list(self.vad_buffer), key=lambda x: x[0]))
        self.highest_seen_ts = max(self.highest_seen_ts, timestamp_sec)
        # 데이터가 추가되면 즉시 처리 시도
        self._try_process_buffers()

    def get_processed_chunk(self):
        """
        [알림] 처리 완료된 오디오 청크를 큐에서 가져옵니다. 없으면 None을 반환합니다.
        """
        if self.output_queue:
            return self.output_queue.popleft()
        return None

    def flush(self):
        """
        [알림] 스트림 종료 시, 버퍼에 남아있는 모든 오디오를 강제로 처리(폴백)하고 반환합니다.
        """
        print("[Reducer] Flushing remaining buffers...")
        # 남은 오디오 강제 처리 (폴백)
        while self.audio_buffer:
            audio_ts, audio_chunk = self.audio_buffer.popleft()
            print(f"  ... Flushing audio chunk at {audio_ts:.2f}s")
            is_speaking_last, vad_conf_last = self.last_vad_state
            processed = self._process_internal(audio_chunk, is_speaking_last, vad_conf_last)
            self.output_queue.append(processed)
        
        # 큐에 남아있는 모든 결과 반환
        flushed_chunks = []
        while self.output_queue:
            flushed_chunks.append(self.output_queue.popleft())
        
        if flushed_chunks:
            return np.concatenate(flushed_chunks)
        return np.array([], dtype=np.float32)

    # -------------------------
    # 내부 버퍼 처리 로직
    # -------------------------
    def _try_process_buffers(self):
        """
        [내부] 코어 동기화 로직.
        처리 전략:
        1) (1순위) audio와 vad가 허용 오차 내에서 매칭되면 즉시 처리
        2) (2순위) VAD가 늦어질 경우: audio가 max_wait_for_vad_sec 이상 기다렸다면
           지연을 막기 위해 'last_vad_state'로 폴백(Fallback) 처리
        3) (3순위) 저지연 모드: 2순위보다 더 공격적으로 폴백(Fallback) 처리
        """
        # 1) 가능한 매칭 처리 (1순위)
        while self.audio_buffer and self.vad_buffer:
            audio_ts, audio_chunk = self.audio_buffer[0]
            vad_ts, is_speaking, vad_conf = self.vad_buffer[0]
            diff = audio_ts - vad_ts

            if abs(diff) <= self.sync_tolerance_sec:
                # Case 1: 매칭 성공
                self.audio_buffer.popleft()
                self.vad_buffer.popleft()
                self.last_vad_state = (is_speaking, vad_conf) # 폴백을 위한 상태 갱신
                processed = self._process_internal(audio_chunk, is_speaking, vad_conf)
                self.output_queue.append(processed)
                continue # 다음 매칭 시도

            # Case 2: VAD가 오디오보다 허용 오차 이상으로 오래된 경우 -> Stale VAD (오래된 VAD)
            if diff > self.sync_tolerance_sec:
                # 이 VAD가 현재 오디오에 비해 너무 이름 -> 오래된 VAD 폐기
                self.vad_buffer.popleft()
                continue # 다음 VAD 확인

            # Case 3: 오디오가 VAD보다 허용 오차 이상으로 오래된 경우 -> 이 오디오에 대한 VAD가 아직 없음
            # 루프를 중단하고 2순위(폴백) 로직으로 넘어감
            break

        # 2) 처리되지 않은 audio에 대해 '대기 허용' 정책 적용 (2, 3순위)
        while self.audio_buffer:
            audio_ts, audio_chunk = self.audio_buffer[0]
            
            # 이 오디오 청크가 VAD를 얼마나 기다렸는지 추정
            waited = max(0.0, self.highest_seen_ts - audio_ts)

            # Case 2: 최대 대기 시간 초과 (2순위)
            if waited >= self.max_wait_for_vad_sec:
                # 충분히 기다렸음 -> 지연을 막기 위해 폴백 처리
                print(f"[Reducer] {audio_ts:.2f}s 오디오 최대 대기 시간 초과. 폴백 처리.")
                self.audio_buffer.popleft()
                is_speaking_last, vad_conf_last = self.last_vad_state
                processed = self._process_internal(audio_chunk, is_speaking_last, vad_conf_last)
                self.output_queue.append(processed)
                continue # 다음 대기 청크 확인

            # Case 3: 저지연 모드 (3순위)
            if self.low_latency_mode:
                # 다음 VAD 타임스탬프 확인 (존재하는 경우)
                if not self.vad_buffer:
                    # VAD 정보가 전혀 없음 -> 마지막 상태를 즉시 사용
                    self.audio_buffer.popleft()
                    is_speaking_last, vad_conf_last = self.last_vad_state
                    processed = self._process_internal(audio_chunk, is_speaking_last, vad_conf_last)
                    self.output_queue.append(processed)
                    continue
                else:
                    # 다음 VAD가 미래에 충분히 멀리 있다면 (> sync_tolerance), 지금 처리
                    next_vad_ts = self.vad_buffer[0][0]
                    if next_vad_ts - audio_ts > self.sync_tolerance_sec:
                        self.audio_buffer.popleft()
                        is_speaking_last, vad_conf_last = self.last_vad_state
                        processed = self._process_internal(audio_chunk, is_speaking_last, vad_conf_last)
                        self.output_queue.append(processed)
                        continue
            
            # 그 외: (Case 4) VAD가 곧 도착할 수 있으므로 더 많은 데이터를 기다리기 위해 중단
            break

    # -------------------------
    # DSP 헬퍼 (STFT/ISTFT, stage0~2)
    # (이하 로직은 잡음 제거 기술 자체에 해당)
    # -------------------------
    def _compute_stft(self, audio):
        if len(audio) < self.n_fft:
            audio = np.pad(audio, (0, self.n_fft - len(audio)))
        S = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length, center=False)
        mag, phase = librosa.magphase(S)
        return mag, phase

    def _reconstruct_audio(self, mag, phase, length):
        try:
            return librosa.istft(mag * phase, hop_length=self.hop_length, center=False, length=length)
        except Exception:
            return librosa.istft(mag * phase, hop_length=self.hop_length, center=False)

    def _update_noise_profile_from_buffer(self):
        if len(self.recent_magnitudes) < self.min_buffer_frames_for_update:
            return
        try:
            buffered = np.hstack(self.recent_magnitudes)
            estimated_floor = np.percentile(buffered, axis=1, q=15, keepdims=True)
            current_mean = float(np.mean(estimated_floor))
            self.noise_floor_history.append(current_mean)

            if self.profile_state == "initializing":
                uf = self.init_update_factor
                self.noise_profile = (1 - uf) * self.noise_profile + uf * estimated_floor
                self.frames_seen += 1
                if self.frames_seen >= self.initial_frames_needed:
                    self.profile_state = "stable"
            elif self.profile_state == "stable":
                uf = self.stable_update_factor
                self.noise_profile = (1 - uf) * self.noise_profile + uf * estimated_floor
                if self.last_noise_mean is not None:
                    delta = abs(current_mean - self.last_noise_mean) / (abs(self.last_noise_mean) + 1e-12)
                    if delta > self.relearn_trigger_ratio:
                        self.profile_state = "relearning"
                        self.frames_seen = 0
            elif self.profile_state == "relearning":
                uf = self.init_update_factor
                self.noise_profile = (1 - uf) * self.noise_profile + uf * estimated_floor
                self.frames_seen += 1
                if self.frames_seen >= self.initial_frames_needed:
                    self.profile_state = "stable"

            self.noise_profile = np.maximum(self.noise_profile, 1e-10)
            self.last_noise_mean = current_mean
        except Exception as e:
            print(f"노이즈 프로파일 업데이트 실패: {e}")
            traceback.print_exc()

    def _correlation_prefilter(self, audio_chunk):
        if len(audio_chunk) == 0:
            return audio_chunk
        mag, phase = self._compute_stft(audio_chunk)
        freq_bins, time_frames = mag.shape
        if not self.recent_magnitudes:
            return self._reconstruct_audio(mag, phase, length=len(audio_chunk))
        ref_stack = np.hstack(self.recent_magnitudes)
        ref_vec = np.mean(ref_stack, axis=1)
        ref_norm = np.linalg.norm(ref_vec) + 1e-12
        ref_vec = ref_vec / ref_norm
        mag_cols = mag
        col_norms = np.linalg.norm(mag_cols, axis=0, keepdims=True) + 1e-12
        mag_norm = mag_cols / col_norms
        corr = np.dot(ref_vec.reshape(1, -1), mag_norm).flatten()
        corr_thresh = self.correlation_threshold
        denom = max(1.0 - corr_thresh, 1e-12)
        corr_mask_t = np.clip((corr - corr_thresh) / denom, 0.0, 1.0)
        corr_mask = np.tile(corr_mask_t.reshape(1, -1), (freq_bins, 1))
        corr_mask_smoothed = nd.gaussian_filter(corr_mask, sigma=(0.0, 1.0))
        filtered_mag = mag * corr_mask_smoothed
        out = self._reconstruct_audio(filtered_mag, phase, length=len(audio_chunk))
        return out

    def _stage1_denoise(self, audio_chunk, is_speaking=None):
        if is_speaking is False:
            return audio_chunk * self.non_speech_gain
        try:
            out = nr.reduce_noise(
                y=audio_chunk,
                sr=self.sample_rate,
                y_noise=self.noise_profile,
                prop_decrease=self.stage1_prop_decrease,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                stationary=self.stationary
            )
            return out
        except Exception as e:
            print(f"Stage1 오류: {e}")
            traceback.print_exc()
            return audio_chunk

    def _stage2_time_smooth(self, audio_chunk, vad_conf):
        min_sigma = 0.5
        max_sigma = 4.0
        sigma = min_sigma + (max_sigma - min_sigma) * np.clip(vad_conf, 0.0, 1.0)
        try:
            smoothed = nd.gaussian_filter1d(audio_chunk.astype(np.float32), sigma=sigma)
        except Exception:
            smoothed = audio_chunk
        blend = np.clip(np.sqrt(vad_conf), 0.0, 1.0)
        out = (1 - blend) * audio_chunk + blend * smoothed
        return out

    # -------------------------
    # 핵심 청크당 처리 (내부)
    # -------------------------
    def _process_internal(self, audio_chunk: np.ndarray, is_speaking: bool = None, vad_confidence: float = None) -> np.ndarray:
        """[내부] 동기화가 완료된 단일 오디오 청크를 VAD 정보와 함께 처리합니다."""
        if audio_chunk is None or len(audio_chunk) == 0:
            return audio_chunk

        vad_conf = 0.0 if vad_confidence is None else float(vad_confidence)
        vad_conf = np.clip(vad_conf, 0.0, 1.0)
        audio_chunk = audio_chunk.astype(np.float32)
        chunk_len = len(audio_chunk)

        # 1. Overlap(중첩)을 인지한 입력
        if self.overlap_len > 0:
            analysis_in = np.concatenate([self.prev_input_tail, audio_chunk])
        else:
            analysis_in = audio_chunk

        # 2. Stage0 (프리필터)
        try:
            prefiltered = self._correlation_prefilter(analysis_in)
        except Exception as e:
            print(f"Stage0 오류: {e}")
            traceback.print_exc()
            prefiltered = analysis_in

        # 3. Stage1 (핵심 잡음 제거)
        try:
            stage1_full = self._stage1_denoise(prefiltered, is_speaking=is_speaking)
        except Exception as e:
            print(f"Stage1 처리 오류: {e}")
            traceback.print_exc()
            stage1_full = prefiltered

        if self.save_stage1_output:
            self.stage1_results_buffer.append(stage1_full)

        # 4. 프로파일 업데이트 (비발화 구간에서만)
        if is_speaking is False:
            try:
                mag, _ = self._compute_stft(analysis_in)
                # n_fft 크기 불일치에 대한 방어 코드
                if mag.shape[0] != self.noise_profile.shape[0]:
                    target = self.noise_profile.shape[0]
                    if mag.shape[0] < target:
                        pad_len = target - mag.shape[0]
                        mag = np.pad(mag, ((0, pad_len), (0, 0)), mode='edge')
                    else:
                        mag = mag[:target, :]
                
                self.recent_magnitudes.append(mag)
                if len(self.recent_magnitudes) > self.max_buffer_frames:
                    self.recent_magnitudes.pop(0)
                self._update_noise_profile_from_buffer()
            except Exception as e:
                print(f"비발화 프로파일 업데이트 오류: {e}")
                traceback.print_exc()

        # 5. Stage2 (스무딩)
        try:
            stage2_full = self._stage2_time_smooth(stage1_full, vad_conf)
        except Exception as e:
            print(f"Stage2 오류: {e}")
            traceback.print_exc()
            stage2_full = stage1_full

        # 6. 중첩(Overlap)을 고려하여 출력 부분 추출
        start_idx = self.overlap_len
        end_idx = start_idx + chunk_len
        if start_idx < 0:
            start_idx = 0
        if end_idx > len(stage2_full):
            end_idx = len(stage2_full)

        output_chunk = stage2_full[start_idx:end_idx]
        
        # 길이 보정
        if len(output_chunk) < chunk_len:
            output_chunk = np.pad(output_chunk, (0, chunk_len - len(output_chunk)))

        # 7. prev_input_tail (이전 입력 꼬리) 갱신 (다음 청크 처리용)
        if self.overlap_len > 0:
            if len(analysis_in) >= self.overlap_len:
                self.prev_input_tail = analysis_in[-self.overlap_len:].copy()
            else:
                tmp = np.zeros(self.overlap_len, dtype=np.float32)
                tmp[-len(analysis_in):] = analysis_in
                self.prev_input_tail = tmp
        else:
            self.prev_input_tail = np.zeros(0, dtype=np.float32)

        return output_chunk.astype(np.float32)

    # -------------------------
    # 저장/로드 유틸리티
    # -------------------------
    def save_noise_profile(self, filepath):
        """[알림] 현재 학습된 노이즈 프로파일을 파일로 저장합니다."""
        try:
            np.save(filepath, self.noise_profile)
            print(f"[알림] 노이즈 프로파일 저장됨: {filepath}")
        except Exception as e:
            print(f"save_noise_profile 오류: {e}")

    def load_noise_profile(self, filepath):
        """[알림] 미리 학습된 노이즈 프로파일을 로드합니다."""
        try:
            if os.path.exists(filepath):
                loaded = np.load(filepath)
                if loaded.shape == self.noise_profile.shape:
                    self.noise_profile = loaded
                    self.is_profile_ready = True
                    print(f"[알림] 노이즈 프로파일 로드됨: {filepath}")
                else:
                    print("[알림] 프로파일 크기 불일치; 무시됨.")
            else:
                print(f"[알림] 프로파일 파일을 찾을 수 없음: {filepath}")
        except Exception as e:
            print(f"노이즈 프로파일 로드 오류: {e}")
            traceback.print_exc()


# ----------------------------------------------------------------------
# ▼▼▼ 모듈 테스트용 헬퍼 함수 ▼▼▼
# (이 함수들은 'if __name__ == "__main__":' 블록에서만 사용됩니다.)
# ----------------------------------------------------------------------

def audio_chunk_generator_with_ts(y, sr, chunk_duration_sec=0.0333):
    """
    (테스트용 헬퍼)
    전체 오디오를 STT나 다른 모듈로 전달하기 위한 청크 스트림으로 만듭니다.
    이 청크 크기(예: 0.0333초)는 비디오 프레임 속도나
    다른 모듈의 처리 단위와 맞추는 것을 시뮬레이션합니다.
    
    Args:
        y (np.ndarray): 전체 오디오 데이터
        sr (int): 샘플 레이트
        chunk_duration_sec (float): 생성할 청크당 시간 (초)
        
    Yields:
        tuple: (오디오 청크, 현재 타임스탬프)
    """
    samples_per_chunk = int(chunk_duration_sec * sr)
    if samples_per_chunk == 0:
        samples_per_chunk = 512 # 최소값 보장 (약 32ms)
        
    current_time_sec = 0.0
    for i in range(0, len(y), samples_per_chunk):
        chunk = y[i:i + samples_per_chunk]
        # 마지막 청크가 너무 짧으면 스킵 (STT 처리 시 문제 방지)
        if len(chunk) < samples_per_chunk / 2 and i + samples_per_chunk < len(y):
             continue
        yield chunk, current_time_sec
        current_time_sec += (len(chunk) / sr) # 실제 청크 길이에 기반한 시간 증가

def get_mock_vad_result_for_test(timestamp):
    """
    (테스트용 헬퍼)
    영상 모듈의 VAD 반환을 시뮬레이션합니다. (가짜 데이터 생성)
    
    (시나리오: 10초 주기, 5초 발화 / 5초 비발화)
    """
    is_speaking = (int(timestamp) % 10) < 5
    vad_confidence = 0.95 if is_speaking else 0.80
    
    return {
        "is_speaking": is_speaking,
        "confidence": vad_confidence,
        "timestamp": timestamp
    }

def run_denoising_test(input_audio_path, output_audio_path):
    # """
    # FinalNoiseReducer 모듈이 STT용 데이터를 잘 생성하는지 테스트하는 함수로 저만 썼습니다....
    # 통합 시에는 고려할 필요 없습니다.
    
    print(f"[테스트] '{input_audio_path}' 파일을 로드하여 잡음 제거 테스트 시작...")
    
    # --- 모듈 초기화 ---
    denoiser = FinalNoiseReducer(
        sample_rate=DEFAULT_SR,
        low_latency_mode=True 
    )
    
    # --- 입력 오디오 로드 ---
    try:
        full_audio_data, sr = librosa.load(input_audio_path, sr=DEFAULT_SR, mono=True)
        print(f"오디오 로드 완료 (SR={sr}Hz, 길이={len(full_audio_data)/sr:.2f}초)")
    except Exception as e:
        print(f"오류: 테스트 오디오 파일 로드 실패. ({input_audio_path}) - {e}")
        return

    # --- 청크 스트리밍 시뮬레이션 (약 30fps 가정) ---
    frame_duration = 1.0 / 30.0 
    
    processed_audio_segments = [] # 디노이징된 조각들을 모으는 리스트
    
    print(f"청크 단위(약 {frame_duration*1000:.1f}ms)로 디노이징 처리 시작...")

    # --- 오디오 청크 루프 (VAD/오디오 주입) ---
    chunk_generator = audio_chunk_generator_with_ts(full_audio_data, DEFAULT_SR, frame_duration)
    
    for audio_chunk, timestamp in chunk_generator:
        
        # --- 가짜 VAD 정보 생성 --- 
        result_dict = get_mock_vad_result_for_test(timestamp)
        is_speaking = result_dict["is_speaking"]
        vad_confidence = result_dict["confidence"]

        # --- API 호출 ---
        denoiser.add_vad_result(is_speaking, vad_confidence, timestamp)
        denoiser.add_audio_chunk(audio_chunk, timestamp)

        # --- API 호출 (결과 인출) ---
        denoised_chunk = denoiser.get_processed_chunk()
        
        # 큐에 1개 이상 쌓일 수 있으므로 while로 비움
        while denoised_chunk is not None:
            processed_audio_segments.append(denoised_chunk)
            denoised_chunk = denoiser.get_processed_chunk()
            
    # --- 스트림 종료 (Flush) ---
    print("스트림 처리 완료. 남은 버퍼(flush) 처리 중...")
    remaining_audio = denoiser.flush()
    if remaining_audio.size > 0:
        processed_audio_segments.append(remaining_audio)

    # --- STT 데이터 생성 (파일 저장) ---
    if not processed_audio_segments:
        print("오류: 처리된 오디오가 없습니다.")
        return
        
    final_processed_audio = np.concatenate(processed_audio_segments)
    
    # 원본과 길이 맞추기 (패딩/절삭으로 인한 미세 오차 보정)
    final_processed_audio = final_processed_audio[:len(full_audio_data)]
    
    try:
        sf.write(output_audio_path, final_processed_audio, DEFAULT_SR)
        print(f"STT용 잡음 제거 오디오가 '{output_audio_path}'에 저장되었습니다.")
    except Exception as e:
        print(f"오류: 최종 오디오 파일 저장 실패 - {e}")


if __name__ == "__main__":
    
    # --- 테스트용 경로 설정 ---
    DATA_DIR = "data"
    INPUT_WAV = os.path.join(DATA_DIR, "input_audio.wav")
    OUTPUT_WAV = os.path.join(DATA_DIR, "denoised_audio_for_stt.wav") 
    
    os.makedirs(DATA_DIR, exist_ok=True)

    # --- 테스트용 오디오 파일 준비 ---
    if not os.path.exists(INPUT_WAV):
        print(f"경고: '{INPUT_WAV}' 파일을 찾을 수 없습니다.")
        print("테스트를 위해 librosa 예제 오디오(trumpet)를 생성합니다.")
        try:
            # 16kHz, 약 5초 길이의 예제 오디오 로드 및 저장
            y, sr = librosa.load(librosa.example('trumpet'), sr=DEFAULT_SR, duration=5.0)
            sf.write(INPUT_WAV, y, sr)
            print(f"테스트용 파일 생성 완료: '{INPUT_WAV}'")
        except Exception as e:
            print(f"오류: librosa 예제 파일 생성 실패. 인터넷 연결을 확인하세요. - {e}")
            exit(1) # 테스트 중단

    run_denoising_test(INPUT_WAV, OUTPUT_WAV)


# STT 모듈과의 통합(호출) 방식 조언

# 필요하실 것 같아 적어둡니다:
# 이 모듈은 VAD(영상 모듈) 정보가 필수적이므로, 
# STT 모듈과 통합하기 위해서는 함수를 직접 호출하는 대신 '스트리밍' 방식으로 호출해야 합니다.

# 생성 (1회): 잡음 제거기 인스턴스를 생성합니다.
    # denoiser = FinalNoiseReducer(sample_rate=16000)

# 처리 (루프): 오디오 청크와 VAD 정보가 생길 때마다 동시에 주입하고, 결과를 즉시 가져와 STT 모듈로 전달합니다.
    # # (오디오/VAD가 생성되는 메인 루프)
    # for (audio_chunk, vad_info) in stream:
    #     # 1. 데이터 주입
    #     denoiser.add_vad_result(vad_info['is_speaking'], vad_info['confidence'], vad_info['timestamp'])
    #     denoiser.add_audio_chunk(audio_chunk, vad_info['timestamp'])

    #     # 2. 결과 인출 (즉시)
    #     processed_chunk = denoiser.get_processed_chunk()
    #     while processed_chunk is not None:
    #         # 3. STT 모듈로 전달
    #         stt_module.process(processed_chunk)
    #         processed_chunk = denoiser.get_processed_chunk()

# 종료 (1회): 루프가 끝나면, 모듈 내부에 남아있던 마지막 오디오를 flush()로 처리합니다.
    # remaining_audio = denoiser.flush()
    # if remaining_audio.size > 0:
    #     stt_module.process(remaining_audio)
    #     stt_module.finalize() # STT 종료