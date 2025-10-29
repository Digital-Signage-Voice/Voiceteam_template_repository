import librosa
import soundfile as sf
import os
import numpy as np
import noisereduce as nr

# --------------------------------------------------------------------------
#  MLNoiseReducer 클래스
# --------------------------------------------------------------------------
class MLNoiseReducer:
    def __init__(self, sample_rate, prop_decrease=1.0, stationary=False):
        self.sample_rate = sample_rate
        # --- 잡음 감소 강도를 클래스 생성 시 결정 ---
        self.prop_decrease = prop_decrease
        self.stationary = stationary
        self.noise_clip = None

    def fit_noise(self, noise_chunk: np.ndarray):
        self.noise_clip = noise_chunk
        print("소음 프로파일 학습 완료.")

    def denoise_audio(self, audio_chunk: np.ndarray, sample_rate: int) -> np.ndarray:
        if self.noise_clip is None:
            raise ValueError("소음 프로파일이 먼저 학습되어야 합니다. fit_noise()를 먼저 호출해주세요.")
        
        if sample_rate != self.sample_rate:
            print(f"경고: 입력된 sample_rate({sample_rate})가 초기화된 값({self.sample_rate})과 다릅니다.")
            
        reduced_chunk = nr.reduce_noise(
            y=audio_chunk,
            sr=sample_rate,
            y_noise=self.noise_clip,
            prop_decrease=self.prop_decrease,
            stationary=self.stationary
        )
        return reduced_chunk

# --------------------------------------------------------------------------
# [헬퍼 함수] 가장 조용한 구간 찾기
# --------------------------------------------------------------------------
def find_quietest_part_indices(full_audio, sr, duration=1.0):
    frame_length = int(duration * sr)
    energies = [np.sum(np.square(frame)) for frame in librosa.util.frame(full_audio, frame_length=frame_length, hop_length=frame_length)]
    
    if not energies:
        return 0, frame_length

    quietest_frame_index = np.argmin(energies)
    start_index = quietest_frame_index * frame_length
    end_index = start_index + frame_length
    
    print(f"가장 조용한 구간(시간: {start_index/sr:.2f}초 ~ {end_index/sr:.2f}초)을 소음으로 학습합니다.")
    return start_index, end_index

# --------------------------------------------------------------------------
#  잡음 제거 강도 조절 
# --------------------------------------------------------------------------
if __name__ == "__main__":
    
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    INTERIM_DATA_DIR = os.path.join(BASE_DIR, 'data', 'interim')
    PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')

    INPUT_WAV = os.path.join(INTERIM_DATA_DIR, 'input_audio.wav')
    
    NOISE_REDUCTION_STRENGTH = 0.8

    output_filename = f'cleaned_audio_v3_strength_{NOISE_REDUCTION_STRENGTH:.1f}.wav'
    OUTPUT_WAV = os.path.join(PROCESSED_DATA_DIR, output_filename)

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    print(f"--- 음성 모듈 (강도 {NOISE_REDUCTION_STRENGTH:.1f}) 데모 시작 ---")

    if not os.path.exists(INPUT_WAV):
        print(f"오류: '{INPUT_WAV}'을 찾을 수 없습니다.")
        print("data/interim/ 폴더에 input_audio.wav 파일을 직접 넣어주세요.")
    else:
        try:
            print("\n[실행] 오디오 파일 로드 중...")
            audio_data, sample_rate = librosa.load(INPUT_WAV, sr=None)
            
            start, end = find_quietest_part_indices(audio_data, sample_rate, duration=1.0)
            noise_clip = audio_data[start:end]
            
            print("ML 기반 잡음 제거 실행 중...")

            reducer = MLNoiseReducer(
                sample_rate=sample_rate,
                prop_decrease=NOISE_REDUCTION_STRENGTH, 
                stationary=False
            )
            
            reducer.fit_noise(noise_chunk=noise_clip)
            
            reduced_noise_audio = reducer.denoise_audio(
                audio_chunk=audio_data,
                sample_rate=sample_rate
            )
            
            sf.write(OUTPUT_WAV, reduced_noise_audio, sample_rate)
            
            print("\n--- 데모 종료 ---")
            print(f"'{INPUT_WAV}'(원본)와 '{OUTPUT_WAV}'(결과물)을 비교해서 들어보세요.")

        except Exception as e:
            print(f"처리 중 오류가 발생했습니다: {e}")

