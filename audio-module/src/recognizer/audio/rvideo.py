import librosa
import soundfile as sf
import os
import numpy as np
import noisereduce as nr
from collections import deque
from scipy.signal import medfilt
import traceback

# --------------------------------------------------------------------------
class FinalNoiseReducer: 
    def __init__(self, sample_rate,
                 noise_window_size=1.0,
                 n_fft=2048, hop_length=512, stationary=False,

                 stage1_prop_decrease=0.98, # 1단계: 상시 적용될 강력한 기본 강도
                 speech_prop_decrease=0.95, # 2단계(발화): 1단계 결과에 적용될 강도

                 apply_smoothing=True,
                 smoothing_kernel_size=3,
                 save_stage1_output=True,
                 initial_noise_db=-40.0
                 ):

        self.sample_rate = sample_rate
        self.noise_window_samples = int(noise_window_size * sample_rate)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.stationary = stationary

        # --- 강도 파라미터 저장 ---
        self.stage1_prop_decrease = stage1_prop_decrease
        self.speech_prop_decrease = speech_prop_decrease
        # ---

        self.apply_smoothing = apply_smoothing
        self.smoothing_kernel_size = smoothing_kernel_size
        self.save_stage1_output = save_stage1_output

        self.recent_magnitudes = []
        self.max_buffer_frames = int(noise_window_size * sample_rate / hop_length)

        num_freq_bins = n_fft // 2 + 1
        initial_noise_power = 10**(initial_noise_db / 10)
        initial_noise_magnitude = np.sqrt(initial_noise_power)
        self.noise_profile = np.full((num_freq_bins, 1), initial_noise_magnitude)
        self.is_profile_ready = True # 초기 프로파일로 시작

        self.stage1_results_buffer = [] if self.save_stage1_output else None
        print(f"Reducer 초기화 완료. 초기 소음 프로파일({initial_noise_db}dB).")

    # --- 소음 플로어 추정 로직 ---
    def _update_noise_profile_from_buffer(self):
        """ 버퍼 분석하여 소음 플로어(magnitude) 업데이트 """
        if len(self.recent_magnitudes) < self.max_buffer_frames // 2: return
        try: # Hstack 오류 방지
            buffered_magnitudes = np.hstack(self.recent_magnitudes)
            estimated_floor = np.percentile(buffered_magnitudes, axis=1, q=15, keepdims=True)
            update_factor = 0.05
            self.noise_profile = (1 - update_factor) * self.noise_profile + update_factor * estimated_floor
            self.noise_profile = np.maximum(self.noise_profile, 1e-10)
        except ValueError: # 가끔 비어있는 버퍼 hstack 시 오류 발생 가능성
             print("경고: 소음 프로파일 업데이트 중 버퍼 오류 발생")
             pass # 일단 무시하고 진행
        print("자동 소음 플로어 업데이트 완료") # 디버깅용

    def _compute_stft(self, audio_chunk):
        stft_result = librosa.stft(audio_chunk, n_fft=self.n_fft, hop_length=self.hop_length)
        return librosa.magphase(stft_result)

    def _reconstruct_audio(self, magnitude, phase):
        reconstructed_stft = magnitude * phase
        return librosa.istft(reconstructed_stft, hop_length=self.hop_length)

    def process_chunk(self, audio_chunk: np.ndarray,
                      is_speaking: bool = None, 
                      ) -> np.ndarray:
        """
        실시간 오디오 청크 처리 (자율 1단계 + is_speaking 기반 분기 및 무음 처리)
        """
        original_length = len(audio_chunk)
        final_output = np.zeros_like(audio_chunk) 

        # --- 스펙트로그램 계산 ---
        try:
            magnitude, phase = self._compute_stft(audio_chunk)
        except Exception as e:
            print(f"STFT 중 오류: {e}"); traceback.print_exc()
            return np.zeros_like(audio_chunk) # 오류 시 안전하게 무음 반환

        # --- 비발화 구간일 경우: 소음 학습만 수행 ---
        if is_speaking is False:
            self.recent_magnitudes.append(magnitude) 
            if len(self.recent_magnitudes) > self.max_buffer_frames: self.recent_magnitudes.pop(0)
            self._update_noise_profile_from_buffer()
            print("  비발화 구간: 소음 학습 수행 + 무음 처리") # 디버깅용
            return final_output # 무음 반환

        # --- 발화 구간 또는 정보 없는 구간 처리 ---
        else: # is_speaking is True or is_speaking is None
            # --- 1단계: 기본 잡음 제거 ---
            stage1_output = audio_chunk
            if self.is_profile_ready:
                try:
                    stage1_output = nr.reduce_noise(
                        y=audio_chunk, 
                        sr=self.sample_rate,
                        y_noise=self.noise_profile,
                        prop_decrease=self.stage1_prop_decrease,
                        n_fft=self.n_fft, hop_length=self.hop_length, stationary=self.stationary
                    )
                    # 길이 맞춤
                    if len(stage1_output) < original_length: stage1_output = np.pad(stage1_output, (0, original_length - len(stage1_output)))
                    elif len(stage1_output) > original_length: stage1_output = stage1_output[:original_length]
                except Exception as e:
                    print(f"1단계 잡음 제거 중 오류 발생: {e}"); traceback.print_exc()
                    stage1_output = audio_chunk # 오류 시 원본
            else: print("경고: 소음 프로파일 준비 안됨") 

            # 1단계 결과 저장
            if self.save_stage1_output:
                self.stage1_results_buffer.append(stage1_output)

            # --- 2단계: (영상 정보=True 시) 추가 미세 조정 및 후처리 ---
            if is_speaking is True and self.is_profile_ready:
                try:
                    print(f"  2단계 처리 (발화, 강도: {self.speech_prop_decrease:.2f})") # 디버깅
                    stage2_output = nr.reduce_noise(
                        y=stage1_output,
                        sr=self.sample_rate,
                        y_noise=self.noise_profile,
                        prop_decrease=self.speech_prop_decrease,
                        n_fft=self.n_fft, hop_length=self.hop_length, stationary=self.stationary
                    )
                    # 길이 맞춤
                    if len(stage2_output) < original_length: stage2_output = np.pad(stage2_output, (0, original_length - len(stage2_output)))
                    elif len(stage2_output) > original_length: stage2_output = stage2_output[:original_length]

                    # 후처리 (스무딩)
                    if self.apply_smoothing:
                        try:
                            mag_proc, phase_proc = self._compute_stft(stage2_output)
                            smoothed_mag = medfilt(mag_proc, kernel_size=(self.smoothing_kernel_size, 1))
                            final_output = self._reconstruct_audio(smoothed_mag, phase_proc)
                            # 길이 맞춤
                            if len(final_output) < original_length: final_output = np.pad(final_output, (0, original_length - len(final_output)))
                            elif len(final_output) > original_length: final_output = final_output[:original_length]
                        except Exception as e: print(f"스무딩 중 오류: {e}"); traceback.print_exc(); final_output = stage2_output # 오류 시 스무딩 전
                    else:
                        final_output = stage2_output

                except Exception as e:
                    print(f"2단계 (발화) 처리 중 오류: {e}"); traceback.print_exc()
                    final_output = stage1_output # 오류 시 1단계 결과
            else:
                 final_output = stage1_output

            return final_output # 최종 처리된 발화 구간 청크 반환

    # --- 1단계 결과 저장 메서드 ---
    def save_result(self, output_path):
        if self.save_stage1_output and self.stage1_results_buffer:
            print(f"1단계 중간 결과 저장 중... ({output_path})")
            try:
                stage1_full_audio = np.concatenate(self.stage1_results_buffer)
                sf.write(output_path, stage1_full_audio, self.sample_rate)
                print("1단계 중간 결과 저장 완료.")
            except Exception as e: print(f"1단계 결과 저장 중 오류: {e}"); traceback.print_exc()
            # ... (try...except 블록 이후) ...
        elif not self.save_stage1_output:
            print("경고: 1단계 결과 저장이 비활성화되어 있습니다.")
        else:
            # 버퍼에 아무 결과도 없을 때 경고 출력
            # 오디오 파일이 너무 짧거나 오류로 처리가 안 된 경우 등도 포함
            print("경고: 처리된 1단계 결과가 없습니다.")

# --- 오디오 청크 제너레이터 ---
def audio_chunk_generator(audio_data, sample_rate, chunk_size_sec=0.1):
    chunk_size_samples = int(chunk_size_sec * sample_rate)
    for i in range(0, len(audio_data), chunk_size_samples):
        chunk = audio_data[i:i + chunk_size_samples]
        if len(chunk) < chunk_size_samples and len(chunk) > 0: yield np.pad(chunk, (0, chunk_size_samples - len(chunk)))
        elif len(chunk) == chunk_size_samples: yield chunk

if __name__ == "__main__":

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    INTERIM_DATA_DIR = os.path.join(BASE_DIR, 'data', 'interim')
    PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
    RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')

    VIDEO_FILE = os.path.join(RAW_DATA_DIR, 'sample_video.mp4')
    INPUT_WAV = os.path.join(INTERIM_DATA_DIR, 'input_audio.wav')
    # --- 파일 이름 정의---
    OUTPUT_STAGE1_WAV = os.path.join(PROCESSED_DATA_DIR, 'cleaned__stage1.wav')
    OUTPUT_FINAL_WAV = os.path.join(PROCESSED_DATA_DIR, 'cleaned__final.wav')

    os.makedirs(INTERIM_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    print("--- 음성 모듈 (V14: 비발화 무음) 데모 시작 ---")
# --- 1단계: 음성 추출 ---
    if not os.path.exists(VIDEO_FILE): print(f"오류: '{VIDEO_FILE}' 없음"); exit()
    if not os.path.exists(INPUT_WAV):
        print(f"\n[STEP 1] '{VIDEO_FILE}'에서 음성 추출 중...")
        FFMPEG_EXE_PATH = r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"; # 사용자 환경 맞게 수정
        if not os.path.exists(FFMPEG_EXE_PATH): FFMPEG_EXE_PATH = "ffmpeg"
        ffmpeg_command = f'"{FFMPEG_EXE_PATH}" -i "{VIDEO_FILE}" -y -vn -acodec pcm_s16le -ar {librosa.get_samplerate(VIDEO_FILE) if os.path.exists(VIDEO_FILE) else 44100} -ac 1 "{INPUT_WAV}"'
        exit_code = os.system(ffmpeg_command);
        if exit_code != 0: print(f"오류: FFmpeg 실패! ({exit_code})"); exit()
        print(f"'{INPUT_WAV}' 파일 생성 완료.")
    else: print(f"\n[STEP 1] 기존 '{INPUT_WAV}' 파일 사용.")

    # --- 2 pase: 실시간 처리 시뮬레이션 (1차만 수행) ---
    try:
        print("\n[STEP 2] 실시간 잡음 제거 시뮬레이션 시작 (Stage 1 Only 모드)...")
        audio_data, sample_rate = librosa.load(INPUT_WAV, sr=None, mono=True)

        # 리듀서 객체 생성 
        reducer = FinalNoiseReducer(
            sample_rate=sample_rate,
            noise_window_size=1.0,
            stage1_prop_decrease=0.98,
            apply_smoothing=False, 
            save_stage1_output=True 
        )

        processed_chunks = []
        total_chunks = (len(audio_data) + int(0.1 * sample_rate) -1) // int(0.1 * sample_rate)
        print(f"오디오를 약 {total_chunks}개의 청크로 나누어 처리합니다...")

        chunk_counter = 0
        for chunk in audio_chunk_generator(audio_data, sample_rate, chunk_size_sec=0.1):
            chunk_counter += 1

            current_time = chunk_counter * 0.1
            sim_is_speaking = False # 어떤 값을 넣든 1단계 결과는 동일하게 조정
            if (5.0 <= current_time < 10.0) or (15.0 <= current_time < 20.0):
                sim_is_speaking = True

            # 코어 로직 호출 (is_speaking 값은 무시됨)
            cleaned_chunk = reducer.process_chunk(
                audio_chunk=chunk,
                is_speaking=sim_is_speaking # 전달은 하지만 내부에서 사용 안 함 => 임의 설정이 까다롭더라고요...
            )
            processed_chunks.append(cleaned_chunk) 

            if chunk_counter % 50 == 0 or chunk_counter == total_chunks:
                profile_status = "학습됨" if reducer.is_profile_ready else f"학습중({len(reducer.recent_magnitudes)}/{reducer.max_buffer_frames})"
                print(f"  청크 {chunk_counter}/{total_chunks} 처리 중... (프로파일: {profile_status})")


        # --- 결과 저장 ---
        reducer.save_result(OUTPUT_FINAL_WAV) 

        print("\n--- 데모 종료 ---")
        print(f"'{INPUT_WAV}'(원본)와 '{OUTPUT_FINAL_WAV}'(1단계 처리 결과)를 비교해서 들어보세요.")

    except Exception as e:
        print(f"처리 중 오류가 발생했습니다: {e}")
        traceback.print_exc()


"""
영상모듈과 통합되면 아래 코드로 진행
"""
# if __name__ == "__main__":

#     BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
#     INTERIM_DATA_DIR = os.path.join(BASE_DIR, 'data', 'interim')
#     PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
#     RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')

#     VIDEO_FILE = os.path.join(RAW_DATA_DIR, 'sample_video.mp4')
#     INPUT_WAV = os.path.join(INTERIM_DATA_DIR, 'input_audio.wav')
#     # --- 파일 이름 정의---
#     OUTPUT_STAGE1_WAV = os.path.join(PROCESSED_DATA_DIR, 'cleaned__stage1.wav')
#     OUTPUT_FINAL_WAV = os.path.join(PROCESSED_DATA_DIR, 'cleaned__final.wav')

#     os.makedirs(INTERIM_DATA_DIR, exist_ok=True)
#     os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

#     print("--- 음성 모듈 (V14: 비발화 무음) 데모 시작 ---")

#     # --- 1단계: 음성 추출 ---
#     if not os.path.exists(VIDEO_FILE): print(f"오류: '{VIDEO_FILE}' 없음"); exit()
#     if not os.path.exists(INPUT_WAV):
#         print(f"\n[STEP 1] '{VIDEO_FILE}'에서 음성 추출 중...")
#         FFMPEG_EXE_PATH = r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"; # 사용자 환경 맞게 수정
#         if not os.path.exists(FFMPEG_EXE_PATH): FFMPEG_EXE_PATH = "ffmpeg"
#         ffmpeg_command = f'"{FFMPEG_EXE_PATH}" -i "{VIDEO_FILE}" -y -vn -acodec pcm_s16le -ar {librosa.get_samplerate(VIDEO_FILE) if os.path.exists(VIDEO_FILE) else 44100} -ac 1 "{INPUT_WAV}"'
#         exit_code = os.system(ffmpeg_command);
#         if exit_code != 0: print(f"오류: FFmpeg 실패! ({exit_code})"); exit()
#         print(f"'{INPUT_WAV}' 파일 생성 완료.")
#     else: print(f"\n[STEP 1] 기존 '{INPUT_WAV}' 파일 사용.")

#     # --- 2단계: 실시간 처리 시뮬레이션 ---
#     try:
#         print("\n[STEP 2] 실시간 잡음 제거 시뮬레이션 시작 (비발화 무음 모드)...")
#         audio_data, sample_rate = librosa.load(INPUT_WAV, sr=None, mono=True)

#         # 리듀서 객체 생성 (비발화 무음 처리)
#         reducer = FinalNoiseReducer(
#             sample_rate=sample_rate,
#             noise_window_size=1.0,
#             stage1_prop_decrease=0.98,
#             speech_prop_decrease=0.95, # 발화 시 약간 약하게
#             apply_smoothing=True,
#             smoothing_kernel_size=3,
#             save_stage1_output=False 
#         )

#         processed_chunks = []
#         total_chunks = (len(audio_data) + int(0.1 * sample_rate) -1) // int(0.1 * sample_rate)
#         print(f"오디오를 약 {total_chunks}개의 청크로 나누어 처리합니다...")

#         chunk_counter = 0
#         for chunk in audio_chunk_generator(audio_data, sample_rate, chunk_size_sec=0.1):
#             chunk_counter += 1
#             current_time = chunk_counter * 0.1

#             # --- [시뮬레이션] 영상 모듈 정보 생성 ---
#             sim_is_speaking = False
#             if (5.0 <= current_time < 10.0) or (15.0 <= current_time < 20.0): # 예시 발화 구간
#                 sim_is_speaking = True
#             else:
#                 sim_is_speaking = False

#             # 코어 로직 호출 (is_speaking 전달)
#             cleaned_chunk = reducer.process_chunk(
#                 audio_chunk=chunk,
#                 is_speaking=sim_is_speaking
#             )
#             processed_chunks.append(cleaned_chunk)

#             if chunk_counter % 50 == 0 or chunk_counter == total_chunks:
#                 status = "발화(처리됨)" if sim_is_speaking else "비발화(무음처리+학습)"
#                 profile_status = "학습됨" if reducer.is_profile_ready else "학습중"
#                 print(f"  청크 {chunk_counter}/{total_chunks} 처리 중... (상태: {status}, 프로파일: {profile_status})")


#         # --- 최종 결과 저장 ---
#         final_cleaned_audio = np.concatenate(processed_chunks)
#         final_cleaned_audio = final_cleaned_audio[:len(audio_data)] 
#         sf.write(OUTPUT_FINAL_WAV, final_cleaned_audio, sample_rate)
#         print(f"\n최종 결과 저장 완료: '{OUTPUT_FINAL_WAV}'")

#         reducer.save_stage1_result(OUTPUT_STAGE1_WAV) # V14 데모에서는 비활성화

#         print("\n--- 데모 종료 ---")
#         print(f"'{INPUT_WAV}'(원본)와 '{OUTPUT_FINAL_WAV}'(최종 결과)를 비교해서 들어보세요.")

#     except Exception as e:
#         print(f"처리 중 오류가 발생했습니다: {e}")
#         traceback.print_exc()
