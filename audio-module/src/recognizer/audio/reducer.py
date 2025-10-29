import librosa
import numpy as np
import soundfile as sf
import os

# --------------------------------------------------------------------------
# [핵심 모듈] AdvancedNoiseReducer 클래스 (성능 개선)
# --------------------------------------------------------------------------
class AdvancedNoiseReducer:
    """
    STFT와 Wiener Filter 기법을 사용하여 오디오 잡음을 제거하는 클래스.
    목소리 손상을 최소화하여 더 자연스러운 결과를 생성함.
    """
    def __init__(self, n_fft=2048, hop_length=512):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.noise_profile = None

    def _compute_stft(self, audio_chunk):
        stft_result = librosa.stft(audio_chunk, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude, phase = librosa.magphase(stft_result)
        return magnitude, phase

    def _reconstruct_audio(self, magnitude, phase):
        reconstructed_stft = magnitude * phase
        audio_chunk = librosa.istft(reconstructed_stft, hop_length=self.hop_length)
        return audio_chunk

    def estimate_noise_profile(self, noise_audio_chunk):
        magnitude, _ = self._compute_stft(noise_audio_chunk)
        self.noise_profile = np.mean(np.square(magnitude), axis=1, keepdims=True)
        print("소음 프로파일(파워) 추정 완료")

    # --- [추가기능] 가장 조용한 구간을 찾는 헬퍼 메서드 ---
    def _find_quietest_part(self, full_audio, sr, duration=1.0):
        """
        오디오 전체에서 가장 에너지가 낮은(조용한) 구간을 찾아 반환합니다.
        """
        frame_length = int(duration * sr)
        
        energies = [np.sum(np.square(frame)) for frame in librosa.util.frame(full_audio, frame_length=frame_length, hop_length=frame_length)]
        
        if not energies:
            return full_audio[:frame_length]

        quietest_frame_index = np.argmin(energies)
        start_index = quietest_frame_index * frame_length
        end_index = start_index + frame_length
        
        print(f"가장 조용한 구간(시간: {start_index/sr:.2f}초 ~ {end_index/sr:.2f}초)을 소음으로 학습합니다.")
        return full_audio[start_index:end_index]

    def process_chunk(self, audio_chunk):
        if self.noise_profile is None:
            raise ValueError("소음 프로파일 선행 요구. estimate_noise_profile()을 먼저 호출해주세요.")
        
        magnitude, phase = self._compute_stft(audio_chunk)

        # --------------------- [수정된 부분: 위너 필터 적용] ---------------------
        # 1. 신호와 잡음의 파워(Power)를 계산
        signal_power = np.square(magnitude)
        noise_power = self.noise_profile
        
        # 2. 신호 대 잡음비 (Signal-to-Noise Ratio, SNR) 추정
        #    0으로 나누는 것을 방지하기 위해 아주 작은 값을 가산
        snr = signal_power / (noise_power + 1e-10)
        
        # 3. 위너 필터 게인(Gain) 계산
        #    SNR이 높을수록 게인은 1에 가까워지고(신호 유지),
        #    SNR이 낮을수록 게인은 0에 가까워짐(잡음 억제).
        gain = snr / (1 + snr)
        
        # 4. 원래 신호의 크기에 게인을 곱하여 잡음 제거
        cleaned_magnitude = magnitude * gain
        # --------------------------------------------------------------------

        cleaned_chunk = self._reconstruct_audio(cleaned_magnitude, phase)
        return cleaned_chunk

    def process_entire_file(self, input_path, output_path, noise_duration=1.0):
        print(f"파일 기반 잡음 제거 시작 (Wiener Filter): '{input_path}'")
        if not os.path.exists(input_path):
            print(f"오류: 입력 파일 '{input_path}'를 찾을 수 없습니다.")
            return
        
        y, sr = librosa.load(input_path, sr=None)
        
        # 가장 조용한 구간을 찾아 소음으로 사용
        noise_sample = self._find_quietest_part(y, sr, duration=noise_duration)
        
        self.estimate_noise_profile(noise_sample)
        cleaned_audio = self.process_chunk(y)
        sf.write(output_path, cleaned_audio, sr)
        print(f"파일 처리 완료! 결과가 '{output_path}'에 저장되었습니다.")


# --------------------------------------------------------------------------
# [데모 실행 부분] - 극도로 단순화된 버전
# --------------------------------------------------------------------------
if __name__ == "__main__":
    
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    INTERIM_DATA_DIR = os.path.join(BASE_DIR, 'data', 'interim')
    PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')

    INPUT_WAV = os.path.join(INTERIM_DATA_DIR, 'input_audio.wav')
    OUTPUT_WAV = os.path.join(PROCESSED_DATA_DIR, 'cleaned_audio_final.wav')

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    print("--- 음성 모듈 (Wiener Filter 버전) 데모 시작 ---")

    # 1단계: 준비된 오디오 파일이 있는지 확인
    if not os.path.exists(INPUT_WAV):
        print(f"오류: '{INPUT_WAV}'을 찾을 수 없습니다.")
        print("data/interim/ 폴더에 input_audio.wav 파일을 직접 넣어주세요.")
    else:
        try:
            # 2단계: 잡음 제거 모듈 바로 실행
            print(f"\n[실행] 잡음 제거 모듈 실행 중...")
            reducer = AdvancedNoiseReducer()
            reducer.process_entire_file(input_path=INPUT_WAV, output_path=OUTPUT_WAV, noise_duration=1.0)
            
            print("\n--- 데모 종료 ---")
            print(f"'{INPUT_WAV}'(원본)와 '{OUTPUT_WAV}'(결과물)을 비교해서 들어보세요.")

        except Exception as e:
            print(f"처리 중 오류가 발생했습니다: {e}")

