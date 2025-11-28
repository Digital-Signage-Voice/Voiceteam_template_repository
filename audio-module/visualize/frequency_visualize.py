import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import os
import sys

def plot_spectrogram(y, sr, title, ax):
    """주어진 오디오 데이터의 스펙트로그램을 그리는 함수"""
    # STFT 파라미터는 reducer 코드와 일치시키는 것이 좋음
    n_fft = 2048
    hop_length = 512
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude, _ = librosa.magphase(D)
    db_spectrogram = librosa.amplitude_to_db(magnitude, ref=np.max)

    img = librosa.display.specshow(db_spectrogram, sr=sr, x_axis='time', y_axis='log', ax=ax, hop_length=hop_length)
    ax.set_title(title)
    return img

# --------------------------------------------------------------------------
# [메인 실행 부분]
# --------------------------------------------------------------------------
if __name__ == "__main__":

    # --- 경로 설정 ---
    VISUALIZE_DIR = os.path.dirname(__file__)
    BASE_DIR = os.path.abspath(os.path.join(VISUALIZE_DIR, '..')) # audio-module 폴더
    INTERIM_DATA_DIR = os.path.join(BASE_DIR, 'data', 'interim')
    PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')

    # --- [V11] 비교할 세 가지 파일 경로 정의 ---
    ORIGINAL_WAV = os.path.join(INTERIM_DATA_DIR, 'input_audio.wav')
    STAGE1_WAV = os.path.join(PROCESSED_DATA_DIR, 'cleaned_audio_v11_stage1_imm.wav') # 1단계 결과
    FINAL_WAV = os.path.join(PROCESSED_DATA_DIR, 'cleaned_audio_v11_final_imm.wav')    # 최종 결과

    print("--- V11 결과 시각화 스크립트 시작 ---")
    print(f"원본 파일: {ORIGINAL_WAV}")
    print(f"1단계 결과 파일: {STAGE1_WAV}")
    print(f"최종 결과 파일: {FINAL_WAV}")

    # --- 파일 로드 ---
    audio_files = {}
    files_to_load = {
        'Original (처리 전)': ORIGINAL_WAV,
        'Stage 1 (즉시 기본 제거)': STAGE1_WAV,
        'Final (2단계 조정 + 후처리)': FINAL_WAV
    }

    loaded_sr = None # 샘플링 레이트 통일 확인용

    for title, path in files_to_load.items():
        if os.path.exists(path):
            try:
                y, sr = librosa.load(path, sr=None)
                audio_files[title] = (y, sr)
                if loaded_sr is None:
                    loaded_sr = sr
                elif loaded_sr != sr:
                    print(f"경고: 파일({os.path.basename(path)})의 샘플링 레이트({sr}Hz)가 이전 파일({loaded_sr}Hz)과 다릅니다.")
            except Exception as e:
                print(f"오류: '{path}' 파일 로드 중 오류 발생: {e}")
        else:
            print(f"경고: 파일({path})을 찾을 수 없습니다. 먼저 실행하세요.")

    # --- 스펙트로그램 그리기 ---
    num_plots = len(audio_files)
    if num_plots == 0:
        print("오류: 표시할 오디오 파일이 없습니다.")
        sys.exit(1)

    fig, axes = plt.subplots(num_plots, 1, sharex=True, figsize=(12, 4 * num_plots))
    if num_plots == 1: axes = [axes] # axes가 배열이 되도록 보장

    plot_index = 0
    images = []
    # 순서를 보장하기 위해 files_to_load 순서대로 그림
    plot_order = ['Original (처리 전)', 'Stage 1 (즉시 기본 제거)', 'Final (2단계 조정 + 후처리)']
    for title in plot_order:
        if title in audio_files:
            y, sr = audio_files[title]
            img = plot_spectrogram(y, sr, title, axes[plot_index])
            images.append(img)
            plot_index += 1

    # 컬러바 추가 (존재하는 이미지 기준)
    if images:
        fig.colorbar(images[-1], ax=axes[:plot_index], format='%+2.0f dB', location='right', aspect=30)

    plt.tight_layout(rect=[0, 0, 0.9, 1]) # 컬러바 공간 확보
    print("그래프 창을 화면에 표시합니다...")
    plt.show() # 그래프 창 띄우기

    print("--- 시각화 스크립트 종료 ---")