import numpy as np
import librosa
import noisereduce as nr
import scipy.ndimage as nd
from collections import deque
from dataclasses import dataclass
from typing import Optional


# -----------------------------
# 설정 관리 (Configuration)
# -----------------------------
@dataclass
class NoiseReducerConfig:
    """잡음 제거 모듈의 하이퍼파라미터 설정"""

    # 오디오 기본 설정
    sample_rate: int = 16000
    n_fft: int = 1024          # 실시간용으로 1024 / 256 조합 권장
    hop_length: int = 256

    # 노이즈 프로파일링 관련 (2단계 / FSM)
    initial_noise_db: float = -40.0
    profile_init_duration_sec: float = 3.0
    min_buffer_frames_for_update: int = 4
    relearn_trigger_ratio: float = 0.30  # 소음 평균 변화 비율 임계치

    # 동기화 / 지연 관련
    sync_tolerance_sec: float = 0.04     # 40ms
    max_wait_for_vad_sec: float = 0.03   # 30ms
    low_latency_mode: bool = True

    # 알고리즘 파라미터 (2단계 기본 값)
    stationary: bool = False
    stage1_prop_decrease: float = 0.9    # 기본 감쇠 강도
    smoothing_kernel_size: int = 3
    vad_confidence_threshold: float = 0.8
    non_speech_gain: float = 0.05        # 비발화 구간 강제 감쇠

    # 자동 튜닝(운영 파라미터)
    min_prop_decrease: float = 0.7       # 깨끗한 환경일 때 하한
    max_prop_decrease: float = 0.98      # 시끄러운 환경일 때 상한
    auto_tune_interval_sec: float = 3.0  # 2~5초 권장 범위 중 3초로 설정

    # 디버깅 / 비상 모드
    bypass_mode: bool = False


class FinalNoiseReducer:
    """
    [Real-time Audio Noise Reducer]

    - 1단계(영상 VAD 모듈)가 준 is_speaking, confidence, timestamp 를 받아
      오디오 스트림과 동기화 후,
    - [논문 내용] 2단계: 발화구간 기본 잡음 제거 + 비발화구간 노이즈 프로파일 학습 + 자동 튜닝
    - [논문 내용] 3단계: confidence 기반 정밀 후처리 (스펙트럼 스무딩, 제한적 복원)

    을 수행
    """

    def __init__(self, config: NoiseReducerConfig = NoiseReducerConfig()):
        self.cfg = config
        self._init_state_variables()
        self._init_buffers()
        self._init_noise_profile()
        print(f"[NoiseReducer] Ready. SR={self.cfg.sample_rate}, LatencyMode={self.cfg.low_latency_mode}")

    # -------------------------
    # 초기 상태/버퍼 설정
    # -------------------------
    def _init_state_variables(self):
        # 노이즈 FSM 상태
        self.profile_state = "initializing"   # initializing -> stable -> relearning
        self.frames_seen = 0
        self.last_noise_mean = None
        self.highest_seen_ts = 0.0

        # 마지막 VAD 상태 (fallback용)
        self.last_vad_state = (False, 0.0)

        # Overlap-Add 용 꼬리 (현재는 사용 X, 확장 여지)
        self.overlap_len = max(0, self.cfg.n_fft - self.cfg.hop_length)
        self.prev_input_tail = np.zeros(self.overlap_len, dtype=np.float32)

        # 런타임 통계 / 자동 튜닝용
        self.stats_window = []       
        self.last_tuning_time = 0.0  # 마지막 자동 튜닝 시각

    def _init_buffers(self):
        self.audio_buffer = deque()   # (timestamp, audio_chunk)
        self.vad_buffer = deque()     # (timestamp, is_speaking, confidence)
        self.output_queue = deque()   # processed audio chunks

        # 노이즈 프로파일: time-domain 버퍼로 유지
        self.noise_buffer = np.zeros(0, dtype=np.float32)

    def _init_noise_profile(self):
        # 노이즈 버퍼 길이(초) 설정
        self.max_noise_sec = 1.0
        self.min_noise_sec = 0.3
        self.max_noise_samples = int(self.cfg.sample_rate * self.max_noise_sec)
        self.min_noise_samples = int(self.cfg.sample_rate * self.min_noise_sec)

        # 초기 소음 플로어 (rms 기준) – cold start 안전장치
        init_power = 10 ** (self.cfg.initial_noise_db / 10.0)
        self.initial_noise_rms = np.sqrt(init_power).astype(np.float32)

    # -------------------------
    # Public API
    # -------------------------
    def add_audio_chunk(self, audio_chunk: np.ndarray, timestamp_sec: float):
        """오디오 청크 입력 (실시간 스트림에서 호출)"""
        self.audio_buffer.append((timestamp_sec,
                                  audio_chunk.astype(np.float32).flatten()))
        self.highest_seen_ts = max(self.highest_seen_ts, timestamp_sec)
        self._process_buffers()

    def add_vad_result(self, is_speaking: bool, vad_confidence: float, timestamp_sec: float):
        """1단계(영상 VAD) 결과 입력"""
        self.vad_buffer.append((timestamp_sec, is_speaking, float(vad_confidence)))

        # 타임스탬프 정렬 (네트워크 지터 방어)
        if len(self.vad_buffer) > 1 and self.vad_buffer[-1][0] < self.vad_buffer[-2][0]:
            self.vad_buffer = deque(sorted(list(self.vad_buffer), key=lambda x: x[0]))

        self.highest_seen_ts = max(self.highest_seen_ts, timestamp_sec)
        self._process_buffers()

    def get_processed_chunk(self) -> Optional[np.ndarray]:
        """처리된 오디오 청크 반환"""
        if self.output_queue:
            return self.output_queue.popleft()
        return None

    def set_bypass(self, enabled: bool):
        """비상용 패스스루 모드"""
        self.cfg.bypass_mode = enabled
        print(f"[NoiseReducer] Bypass mode: {enabled}")

    def flush(self) -> np.ndarray:
        """남은 버퍼 강제 처리"""
        flushed_chunks = []

        while self.audio_buffer:
            ts, chunk = self.audio_buffer.popleft()
            is_speaking, conf = self.last_vad_state
            processed = self._denoise_core(chunk, is_speaking, conf)
            flushed_chunks.append(processed)

        while self.output_queue:
            flushed_chunks.append(self.output_queue.popleft())

        return np.concatenate(flushed_chunks) if flushed_chunks else np.array([], dtype=np.float32)

    # -------------------------
    # Sync / 매칭 로직 (2.2의 타임스탬프 동기화 부분)
    # -------------------------
    def _process_buffers(self):
        """
        시각 정보와 오디오 버퍼 타임스탬프를 매칭해서
        허용 오차 내의 청크만 유효 처리.
        """
        import time

        while self.audio_buffer:
            audio_ts, audio_chunk = self.audio_buffer[0]

            # 1) Bypass 모드면 그대로 통과
            if self.cfg.bypass_mode:
                self.audio_buffer.popleft()
                self.output_queue.append(audio_chunk)
                continue

            # 2) VAD가 아직 안 온 상태
            if not self.vad_buffer:
                waited = self.highest_seen_ts - audio_ts
                if waited >= self.cfg.max_wait_for_vad_sec or self.cfg.low_latency_mode:
                    # 저지연 모드에서는 일정 시간 이상 기다리지 않고 Fallback 처리
                    self._fallback_processing()
                else:
                    break
                continue

            vad_ts, is_speaking, vad_conf = self.vad_buffer[0]
            diff = audio_ts - vad_ts

            if abs(diff) <= self.cfg.sync_tolerance_sec:
                # [Case 1] 동기화 성공
                self.audio_buffer.popleft()
                self.vad_buffer.popleft()
                self.last_vad_state = (is_speaking, vad_conf)

                processed = self._denoise_core(audio_chunk, is_speaking, vad_conf)
                self.output_queue.append(processed)

            elif diff > self.cfg.sync_tolerance_sec:
                # [Case 2] VAD가 너무 과거 – 폐기
                self.vad_buffer.popleft()

            else:
                # [Case 3] 해당 오디오에 맞는 VAD가 아직 안 옴
                waited = self.highest_seen_ts - audio_ts
                if waited >= self.cfg.max_wait_for_vad_sec:
                    self._fallback_processing()
                else:
                    break

    def _fallback_processing(self):
        """VAD 없이 마지막 상태로 처리 (저지연 방어)"""
        if not self.audio_buffer:
            return
        _, audio_chunk = self.audio_buffer.popleft()
        is_speaking, conf = self.last_vad_state
        processed = self._denoise_core(audio_chunk, is_speaking, conf)
        self.output_queue.append(processed)

    # -------------------------
    # 노이즈 프로파일링 / FSM 
    # -------------------------
    def _accumulate_noise_chunk(self, audio_chunk: np.ndarray):
        """
        비발화(non-speech) 구간 오디오를 time-domain 노이즈 버퍼로 축적.
        동시에 노이즈 평균값을 이용해 FSM(initializing/stable/relearning) 상태 전이.
        """
        if audio_chunk is None or len(audio_chunk) == 0:
            return

        chunk = audio_chunk.astype(np.float32).flatten()

        # time-domain 노이즈 버퍼 갱신
        self.noise_buffer = np.concatenate([self.noise_buffer, chunk])
        if len(self.noise_buffer) > self.max_noise_samples:
            self.noise_buffer = self.noise_buffer[-self.max_noise_samples:]

        self.frames_seen += 1
        current_mean = float(np.mean(np.abs(chunk)))

        if self.last_noise_mean is None:
            self.last_noise_mean = current_mean
            return

        # FSM 상태 전이
        if self.profile_state == "initializing":
            # 최소 N프레임 이상 비발화 데이터 확보 시 안정 단계로
            if self.frames_seen >= self.cfg.min_buffer_frames_for_update:
                self.profile_state = "stable"

        elif self.profile_state == "stable":
            # 소음 평균 변화 비율이 임계치 초과 → 재학습 상태 전환
            if self.last_noise_mean > 0:
                change_ratio = abs(current_mean - self.last_noise_mean) / self.last_noise_mean
                if change_ratio > self.cfg.relearn_trigger_ratio:
                    self.profile_state = "relearning"
                    # 새 환경에 맞게 버퍼 리셋 후 재수렴
                    self.noise_buffer = chunk.copy()
                    self.frames_seen = 1

        elif self.profile_state == "relearning":
            # 일정 프레임 이상 모이면 다시 stable 로 전환
            if self.frames_seen >= self.cfg.min_buffer_frames_for_update:
                self.profile_state = "stable"

        self.last_noise_mean = current_mean

    def _inflight_noise_learning(self, audio_chunk: np.ndarray):
        """
        발화 구간 내부에서 상대적으로 에너지가 낮은 성분을 소음으로 추정하는
        in-flight 프로파일링 (논문 2.2 뒷부분 내용).
        비발화 샘플이 충분치 않을 때만 동작.
        """
        if len(self.noise_buffer) >= self.min_noise_samples:
            return

        x = audio_chunk.astype(np.float32).flatten()
        if len(x) == 0:
            return

        rms = float(np.sqrt(np.mean(x ** 2)) + 1e-8)
        if rms == 0:
            return

        # 전체 RMS의 50% 미만인 구간을 noise candidate 로 사용 (간단한 휴리스틱)
        mask = np.abs(x) < 0.5 * rms
        candidates = x[mask]
        if candidates.size == 0:
            return

        self.noise_buffer = np.concatenate([self.noise_buffer, candidates])
        if len(self.noise_buffer) > self.max_noise_samples:
            self.noise_buffer = self.noise_buffer[-self.max_noise_samples:]

    # -------------------------
    # 런타임 통계 / 자동 튜닝
    # -------------------------
    def _update_runtime_stats(self, speech_chunk: np.ndarray):
        """
        발화 RMS, noised RMS를 이용해 간이 SNR를 추정하고
        일정 주기마다 prop_decrease를 자동 튜닝.
        """
        import time
        now = time.time()

        speech = speech_chunk.astype(np.float32)
        speech_rms = float(np.sqrt(np.mean(speech ** 2)) + 1e-8)

        if len(self.noise_buffer) > 0:
            noise_rms = float(np.sqrt(np.mean(self.noise_buffer ** 2)) + 1e-8)
        else:
            noise_rms = self.initial_noise_rms

        snr_db = 20.0 * np.log10(speech_rms / noise_rms + 1e-8)

        self.stats_window.append({"t": now, "snr_db": snr_db})
        # 최근 10초만 유지
        self.stats_window = [s for s in self.stats_window if now - s["t"] <= 10.0]

        # 자동 튜닝 주기가 지났으면 prop_decrease 조정
        if now - self.last_tuning_time >= self.cfg.auto_tune_interval_sec:
            self._auto_tune_hyperparams()
            self.last_tuning_time = now

    def _auto_tune_hyperparams(self):
        """median SNR 기반의 간단한 3단계 정책으로 prop_decrease 자동 튜닝"""
        if not self.stats_window:
            return

        snrs = [s["snr_db"] for s in self.stats_window]
        median_snr = float(np.median(snrs))

        # e.g.) SNR < 5dB: 시끄러운 환경 → 강한 억제
        #     SNR > 15dB: 비교적 깨끗 → 음질 우선
        if median_snr < 5.0:
            target = self.cfg.max_prop_decrease
        elif median_snr > 15.0:
            target = self.cfg.min_prop_decrease
        else:
            # 5~15dB 구간에서는 선형 보간
            ratio = (15.0 - median_snr) / 10.0
            target = self.cfg.min_prop_decrease + ratio * (
                self.cfg.max_prop_decrease - self.cfg.min_prop_decrease
            )

        self.cfg.stage1_prop_decrease = float(
            np.clip(target, self.cfg.min_prop_decrease, self.cfg.max_prop_decrease)
        )

    # -------------------------
    # DSP Core
    # -------------------------
    def _denoise_core(self, audio_chunk: np.ndarray,
                      is_speaking: bool, vad_conf: float) -> np.ndarray:
        """2단계 + 3단계 전체 플로우"""

        if audio_chunk is None or len(audio_chunk) == 0:
            return audio_chunk

        x = audio_chunk.astype(np.float32).flatten()

        # ===== 2단계: 비발화 구간 처리 (노이즈 학습 + 강제 감쇠) =====
        if not is_speaking:
            # 비발화 구간은 노이즈 프로파일 학습용 데이터로 사용
            self._accumulate_noise_chunk(x)
            # STT 오류 방지: 강하게 감쇠시켜 전달
            return x * self.cfg.non_speech_gain

        # ===== 2단계: 발화 구간 기본 잡음 제거 =====
        # 비발화 샘플이 부족하면 in-flight noise profiling 수행
        self._inflight_noise_learning(x)
        # 런타임 통계 업데이트 (SNR 기반 자동 튜닝)
        self._update_runtime_stats(x)

        # Stage1: 기본 노이즈 감소
        stage1 = self._stage1_denoise(x, vad_conf)

        # ===== 3단계: confidence 기반 정밀 후처리 =====
        out = self._stage2_confidence_refine(stage1, vad_conf)
        return out

    # -------------------------
    # 2단계 내부: Stage1 NR
    # -------------------------
    def _current_prop_decrease(self, vad_conf: float) -> float:
        """
        자동 튜닝된 stage1_prop_decrease를 기반으로
        confidence 에 따라 소폭 가감.
        """
        base = self.cfg.stage1_prop_decrease
        # conf >= 0.8 에서만 조금 더 aggressive 하게
        alpha = float(np.clip((vad_conf - 0.8) / 0.2, 0.0, 1.0))
        pd = 0.8 + alpha * (base - 0.8)
        return float(np.clip(pd, self.cfg.min_prop_decrease, self.cfg.max_prop_decrease))

    def _stage1_denoise(self, x: np.ndarray, vad_conf: float) -> np.ndarray:
        """Stage1: noisereduce 기반 기본 잡음 제거"""
        prop = self._current_prop_decrease(vad_conf)

        try:
            use_noise = len(self.noise_buffer) >= self.min_noise_samples

            if use_noise:
                # 비발화/인플라이트 학습으로 축적한 time-domain 노이즈 버퍼 사용
                y = nr.reduce_noise(
                    y=x,
                    sr=self.cfg.sample_rate,
                    y_noise=self.noise_buffer,
                    prop_decrease=prop,
                    stationary=self.cfg.stationary,
                )
            else:
                # 아직 노이즈 버퍼가 충분치 않으면 내부 추정에 맡김
                y = nr.reduce_noise(
                    y=x,
                    sr=self.cfg.sample_rate,
                    prop_decrease=prop,
                    stationary=self.cfg.stationary,
                )

            return y.astype(np.float32)
        except Exception as e:
            print(f"[ERROR] Stage1 reduce_noise fail: {e}")
            return x

    # -------------------------
    # 3단계 내부: Confidence 기반 정밀 후처리
    # -------------------------
    def _compute_sigma(self, vad_conf: float) -> float:
        """
        - confidence ↑ → sigma ↓ (선명하게)
        - confidence ↓ → sigma ↑ (더 부드럽게 smoothing)
        """
        conf = float(np.clip(vad_conf, 0.0, 1.0))
        sigma_max = 1.5
        sigma_min = 0.0
        sigma = sigma_max * (1.0 - conf)
        return float(np.clip(sigma, sigma_min, sigma_max))

    def _stage2_confidence_refine(self, stage1: np.ndarray, vad_conf: float) -> np.ndarray:
        """
        3단계: confidence 기반 정밀 잡음 제거
        - confidence < threshold: stage1_output 그대로 패스
        - confidence >= threshold:
            (1) STFT
            (2) 스펙트럼 스무딩 (median + Gaussian)
            (3) 제한적 gain 보정
            (4) ISTFT
        """
        if vad_conf < self.cfg.vad_confidence_threshold:
            # 임계값 미만일 경우 stage1_output 그대로 전달
            return stage1

        try:
            # (1) STFT
            stft = librosa.stft(stage1,
                                n_fft=self.cfg.n_fft,
                                hop_length=self.cfg.hop_length,
                                center=False)
            mag, phase = np.abs(stft), np.angle(stft)

            # (2) 스펙트럼 스무딩
            mag_smooth = self._apply_spectral_smoothing(mag, vad_conf)

            # (3) musical noise/과도 억제 보정:
            #     stage1 대비 ± 제한된 범위에서만 gain 조정
            gain = mag_smooth / (mag + 1e-8)
            # confidence가 높을수록 보정폭을 조금 더 허용할 수도 있음 (단순 버전)
            gain = np.clip(gain, 0.5, 1.5)

            mag_refined = mag * gain
            stft_refined = mag_refined * np.exp(1j * phase)

            # (4) ISTFT
            y = librosa.istft(stft_refined,
                              hop_length=self.cfg.hop_length,
                              length=len(stage1),
                              center=False)
            return y.astype(np.float32)

        except Exception as e:
            print(f"[WARN] Stage2 refine fail: {e}")
            return stage1

    def _apply_spectral_smoothing(self, mag: np.ndarray, vad_conf: float) -> np.ndarray:
        """
        스펙트럼 스무딩:
        - 2D median filter (freq x time)로 급격한 홀/스파이크 제거
        - confidence 기반 Gaussian 필터로 추가 스무딩
        """
        # 2D median filter (3 x K)
        if self.cfg.smoothing_kernel_size > 1:
            k = self.cfg.smoothing_kernel_size
            mag = nd.median_filter(mag, size=(3, k))

        # Gaussian smoothing (freq,time) – sigma는 confidence에 반비례
        sigma_t = self._compute_sigma(vad_conf)
        if sigma_t > 0.0:
            # 주파수축은 sigma=1.0, 시간축은 sigma_t
            mag = nd.gaussian_filter(mag, sigma=(1.0, sigma_t))

        return mag
