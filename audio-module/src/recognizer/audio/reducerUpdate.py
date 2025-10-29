import librosa
import soundfile as sf
import os
import numpy as np
import time

# --------------------------------------------------------------------------
#  SpectralNoiseReducer 클래스 (STFT/ISTFT 직접 구현)
# --------------------------------------------------------------------------
class SpectralNoiseReducer:
    def __init__(self, sample_rate, n_fft=2048, hop_length=512, prop_decrease=1.0, initial_noise_db=-40):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.prop_decrease = prop_decrease # 잡음 감소 강도
        
        self.n_freq_bins = n_fft // 2 + 1
        
        # --- 초기 소음 프로파일 설정 (손님 요청: "강한 걸로 교체") ---
        # 묵음(-inf)이 아닌, -40dB 정도의 약한 잡음이 깔려있다고 가정
        # (이전 V8의 -80dB보다 훨씬 '강한' 초기값)
        initial_noise_power = 10**(initial_noise_db / 10)
        initial_noise_magnitude = np.sqrt(initial_noise_power)
        
        # 모든 주파수 대역에 균일한 초기 프로파일 생성
        self.noise_profile = np.full((self.n_freq_bins,), initial_noise_magnitude)
        self.is_profile_ready = True
        print(f"  - Reducer 초기화: {self.n_fft} N_FFT, {self.hop_length} Hop Length")
        print(f"  - 초기 소음 프로파일: {initial_noise_db} dB 기준으로 설정됨")

    def _stft(self, y):
        """STFT를 계산하고 Magnitude와 Phase를 반환"""
        S_complex = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        S_mag, S_phase = librosa.magphase(S_complex)
        return S_mag, S_phase

    def _istft(self, S_mag, S_phase, length):
        """Magnitude와 Phase를 결합하여 ISTFT 수행"""
        S_complex = S_mag * S_phase
        y_denoised = librosa.istft(S_complex, hop_length=self.hop_length, length=length)
        return y_denoised

    def update_noise_profile(self, noise_chunk: np.ndarray):
        """
        VAD에 의해 '소음'으로 판단된 청크를 사용해 소음 프로파일을 갱신
        (Spectral Smoothing 적용)
        """
        noise_mag, _ = self._stft(noise_chunk)
        
        # STFT는 (주파수, 시간) 2D 배열이므로 시간 축에 대해 평균
        noise_mag_mean = np.mean(noise_mag, axis=1)
        
        # 지수 이동 평균 (Exponential Moving Average)으로 부드럽게 갱신
        self.noise_profile = (0.8 * self.noise_profile) + (0.2 * noise_mag_mean)
        # print("  (Noise Profile Updated...)") # (디버그용)

    def denoise_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
        """
        VAD에 의해 '음성'으로 판단된 청크의 잡음을 제거합니다.
        """
        if not self.is_profile_ready:
            return audio_chunk # 프로파일 없으면 원본 반환

        original_length = len(audio_chunk)
        audio_mag, audio_phase = self._stft(audio_chunk)

        # STFT 결과 (n_bins, n_frames)와 프로파일 (n_bins,)을 브로드캐스팅하기 위해
        # noise_profile을 (n_bins, 1)로 차원 확장
        noise_profile_expanded = np.expand_dims(self.noise_profile, axis=1)

        # --- 핵심: 스펙트럼 차감 (Spectral Subtraction) ---
        # (음성 크기 - (소음 크기 * 감소 강도))
        mag_denoised = audio_mag - (noise_profile_expanded * self.prop_decrease)
        
        # 음수가 된 값은 0으로 클리핑 (음악적 소음 방지)
        mag_denoised = np.maximum(0, mag_denoised)

        # ISTFT로 오디오 복원
        denoised_chunk = self._istft(mag_denoised, audio_phase, length=original_length)
        
        return denoised_chunk

# --------------------------------------------------------------------------
# 적응형 스펙트럼 차감 (Adaptive VAD + Overlap-Save)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    
    # --- 경로 설정 ---
    try:
        BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    except NameError:
        BASE_DIR = os.path.abspath('.')
        
    INTERIM_DATA_DIR = os.path.join(BASE_DIR, 'data', 'interim')
    PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
    # -----------------

    INPUT_WAV = os.path.join(INTERIM_DATA_DIR, 'input_audio.wav')
    
    # --- 설정값 ---
    NOISE_REDUCTION_STRENGTH = 1.0   # 잡음 감소 강도 (1.0 = 100%)
    CHUNK_DURATION_MS = 100          # 100ms (Hop Size)
    
    # VAD 임계값 (RMSE 기준). 이 값보다 낮으면 '소음', 높으면 '음성'
    # 0.005 ~ 0.01 정도가 일반적. 환경에 따라 튜닝 필요
    VAD_ENERGY_THRESHOLD = 0.015

    # STFT 파라미터 (Overlap-Save를 위해 PROCESS_SIZE와 HOP_SIZE 계산)
    N_FFT = 2048
    HOP_LENGTH = N_FFT // 4 # (512)
    # -----------------

    # v6: VAD 적용
    output_filename = f'cleaned_audio_v6_adaptive_vad_{VAD_ENERGY_THRESHOLD:.3f}.wav'
    OUTPUT_WAV = os.path.join(PROCESSED_DATA_DIR, output_filename)

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    print(f"--- 실시간 청크 처리 V6 (Adaptive VAD, 강도 {NOISE_REDUCTION_STRENGTH:.1f}) 데모 시작 ---")

    if not os.path.exists(INPUT_WAV):
        print(f"오류: '{INPUT_WAV}'을 찾을 수 없습니다.")
        print(f"현재 폴더({BASE_DIR}) 내에 'data/interim/input_audio.wav' 파일이 있는지 확인해주세요.")
    else:
        try:
            print(f"\n[실행] '{INPUT_WAV}' 파일 정보 읽기...")
            info = sf.info(INPUT_WAV)
            sample_rate = info.samplerate
            channels = info.channels
            
            # --- Overlap-Save 로직 ---
            # HOP_SIZE: 한 번에 읽고/쓸 청크 크기 (100ms)
            HOP_SIZE = int(sample_rate * (CHUNK_DURATION_MS / 1000.0))
            # PROCESS_SIZE: STFT 처리를 위해 겹칠 크기 (2 * HOP_SIZE = 200ms)
            PROCESS_SIZE = HOP_SIZE * 2
            # --------------------------------

            print(f"  - Sample Rate: {sample_rate} Hz, Channels: {channels}")
            print(f"  - Hop Size (저장 단위): {HOP_SIZE} samples ({CHUNK_DURATION_MS} ms)")
            print(f"  - Process Size (처리 단위): {PROCESS_SIZE} samples ({CHUNK_DURATION_MS * 2} ms)")
            print(f"  - VAD 임계값 (에너지): {VAD_ENERGY_THRESHOLD}\n")

            # 리듀서 초기화
            reducer = SpectralNoiseReducer(
                sample_rate=sample_rate,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                prop_decrease=NOISE_REDUCTION_STRENGTH
            )

            main_buffer = np.array([], dtype=np.float32)
            start_time = time.time()
            total_chunks_processed = 0

            with sf.SoundFile(OUTPUT_WAV, 'w', samplerate=sample_rate, channels=1) as output_file:
                
                # 100ms (HOP_SIZE)씩 
                for i, chunk in enumerate(sf.blocks(INPUT_WAV, blocksize=HOP_SIZE, dtype='float32')):
                    
                    if chunk.ndim > 1:
                        chunk_mono = np.mean(chunk, axis=1)
                    else:
                        chunk_mono = chunk
                    
                    main_buffer = np.concatenate((main_buffer, chunk_mono))

                    # 버퍼에 처리할 만큼(200ms) 데이터가 쌓였는지 확인
                    while len(main_buffer) >= PROCESS_SIZE:
                        # 200ms 분량의 청크를 꺼냄
                        chunk_to_process = main_buffer[:PROCESS_SIZE]
                        
                        # --- VAD (음성 활동 감지) ---
                        # RMSE(Root Mean Square Error)로 에너지 계산
                        energy = np.sqrt(np.mean(np.square(chunk_to_process)))

                        output_chunk = None # 저장할 오디오 조각

                        if energy < VAD_ENERGY_THRESHOLD:
                            # [판단: 소음/묵음]
                            # 소음 프로파일을 갱신합니다.
                            reducer.update_noise_profile(chunk_to_process)
                            # 소음 구간은 묵음 처리
                            output_chunk = chunk_to_process[:HOP_SIZE] * 0.1 # 소리를 90% 줄임
                        
                        else:
                            # [판단: 음성]
                            # 잡음을 제거합니다.
                            denoised_chunk = reducer.denoise_chunk(chunk_to_process)
                            # 결과물(200ms)에서 앞 100ms(HOP_SIZE)만 저장
                            output_chunk = denoised_chunk[:HOP_SIZE]

                        # -----------------------------
                        
                        output_file.write(output_chunk)
                        
                        # 버퍼에서 100ms(HOP_SIZE)만큼만 제거
                        main_buffer = main_buffer[HOP_SIZE:]

                        total_chunks_processed += 1
                        if total_chunks_processed % 100 == 0: # 10초마다
                            print(f"  ...처리 중: {total_chunks_processed * CHUNK_DURATION_MS / 1000.0:.1f}초 경과 (현재 에너지: {energy:.4f})")

                # 3. 루프 종료 후, 버퍼에 남은 찌꺼기 처리
                if len(main_buffer) > 0:
                    print(f"  ...파일 끝 도달. 남은 {len(main_buffer)/sample_rate:.2f}초 분량 처리 중...")
                    pad_len = PROCESS_SIZE - len(main_buffer)
                    final_chunk = np.pad(main_buffer, (0, pad_len), 'constant')
                    
                    denoised_chunk = reducer.denoise_chunk(final_chunk)
                    output_file.write(denoised_chunk[:len(main_buffer)])

            end_time = time.time()
            print("\n--- 데모 종료 ---")
            print(f"총 처리 시간: {end_time - start_time:.2f}초")
            print(f"'{INPUT_WAV}'(원본)와 '{OUTPUT_WAV}'(결과물)을 비교해서 들어보세요.")
            print(f"TIP: 잡음이 너무 많이 남으면 VAD_ENERGY_THRESHOLD 값을 높이고({VAD_ENERGY_THRESHOLD*1.5:.3f}),")
            print(f"     목소리가 먹히면 VAD_ENERGY_THRESHOLD 값을 낮춰보세요({VAD_ENERGY_THRESHOLD*0.5:.3f}).")


        except Exception as e:
            print(f"처리 중 오류가 발생했습니다: {e}")