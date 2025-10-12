import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_spectrograms(original_path, processed_path):
    """
    원본 오디오와 처리된 오디오의 스펙트로그램을 나란히 그려서 비교합니다.
    """
    if not os.path.exists(original_path) or not os.path.exists(processed_path):
        print("오류: 비교할 원본 또는 처리된 파일이 없습니다.")
        print(f"'{original_path}' 또는 '{processed_path}' 경로를 확인해주세요.")
        return

    # 1. 오디오 파일 로드
    y_orig, sr_orig = librosa.load(original_path)
    y_proc, sr_proc = librosa.load(processed_path)

    # 2. 각 오디오의 스펙트로그램 계산
    D_orig = librosa.stft(y_orig)
    D_proc = librosa.stft(y_proc)
    
    db_orig = librosa.amplitude_to_db(np.abs(D_orig), ref=np.max)
    db_proc = librosa.amplitude_to_db(np.abs(D_proc), ref=np.max)

    # 3. Matplotlib을 이용한 시각화
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(12, 10))

    # 원본 스펙트로그램
    img1 = librosa.display.specshow(db_orig, sr=sr_orig, x_axis='time', y_axis='log', ax=ax1)
    ax1.set_title('Original Spectrogram (처리 전)')
    fig.colorbar(img1, ax=ax1, format='%+2.0f dB')

    # 처리된 스펙트로그램
    img2 = librosa.display.specshow(db_proc, sr=sr_proc, x_axis='time', y_axis='log', ax=ax2)
    ax2.set_title('Processed Spectrogram (처리 후)')
    fig.colorbar(img2, ax=ax2, format='%+2.0f dB')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    INTERIM_DATA_DIR = os.path.join(BASE_DIR, 'data', 'interim')
    PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')

    INPUT_WAV = os.path.join(INTERIM_DATA_DIR, 'input_audio.wav')
    OUTPUT_WAV = os.path.join(PROCESSED_DATA_DIR, 'cleaned_audio_final.wav')
    
    print("--- 잡음 제거 결과 시각화 ---")
    plot_spectrograms(original_path=INPUT_WAV, processed_path=OUTPUT_WAV)
