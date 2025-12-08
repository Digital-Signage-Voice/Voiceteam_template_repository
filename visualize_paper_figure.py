import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

# 파일 경로
FILE_PATHS = {
    "original": "data/test_original.wav",   # 1. 원본
    "baseline": "data/test_baseline.wav",   # 2. 기존 기술 (VAD 없음)
    "stage1":   "data/test_stage1.wav",     # 3. 1단계 
    "stage2":   "data/test_stage2.wav"      # 4. 최종 결과
}

TARGET_SR = 16000  # 샘플링 레이트 (논문 기준 16kHz)

def load_and_trim_audio(path, target_sr):
    """오디오 파일을 불러오고 데이터가 없는 경우를 방지"""
    if not os.path.exists(path):
        print(f"⚠️ 경고: 파일을 찾을 수 없습니다 -> {path}")
        # 파일 없으면 빈 깡통 리턴 (에러 방지용)
        return np.zeros(int(target_sr * 4)), target_sr
    
    y, sr = librosa.load(path, sr=target_sr)
    return y, sr

def draw_real_spectrograms(file_paths):
    """
    4개의 실제 오디오 파일을 불러와 논문용 스펙트로그램을 그립니다.
    """
    # 1. 데이터 로드
    print("오디오 파일 로딩 중...")
    data = {}
    min_len = float('inf')

    # 모든 파일을 불러옵니다.
    for key, path in file_paths.items():
        y, sr = load_and_trim_audio(path, TARGET_SR)
        data[key] = y
        # 시각화 싱크를 맞추기 위해 가장 짧은 길이를 찾습니다.
        if len(y) > 0:
            min_len = min(min_len, len(y))

    # 데이터가 하나도 없으면 중단
    if min_len == float('inf') or min_len == 0:
        print("유효한 오디오 데이터가 없습니다. 경로를 확인해주세요.")
        return

    # 2. 길이 동기화 (모든 그래프의 X축 길이를 똑같이 맞춤)
    for key in data:
        data[key] = data[key][:min_len]

    # 3. 플롯 설정 
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True, constrained_layout=True)
    
    # 그래프 그리기 헬퍼 함수
    def plot_spec(ax, y, title, label_char):
        # STFT 변환 (스펙트로그램)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        
        # 그리기
        img = librosa.display.specshow(D, sr=TARGET_SR, x_axis='time', y_axis='hz', 
                                     ax=ax, cmap='magma') # 'inferno'나 'magma' 추천
        
        # 라벨 및 스타일링
        ax.set_title(title, fontsize=13, fontweight='bold', loc='left')
        ax.set_ylabel("Freq (Hz)", fontsize=10)
        ax.set_ylim(0, 8000) # 사람 목소리 대역 위주로 (필요시 조정)
        
        # (a), (b), (c), (d) 박스 표기
        ax.text(0.015, 0.92, f"({label_char})", transform=ax.transAxes, 
                fontsize=14, fontweight='bold', color='white',
                bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.2'))
        return img

    # 순서대로 그리기
    # (a) Original
    plot_spec(axes[0], data['original'], "Original Input (Noisy Environment)", "a")
    
    # (b) Baseline
    plot_spec(axes[1], data['baseline'], "Baseline (Audio-only VAD / Traditional)", "b")
    
    # (c) Stage 1
    plot_spec(axes[2], data['stage1'], "Stage 1: Visual Gating + 1st Denoising", "c")
    
    # (d) Stage 2
    img = plot_spec(axes[3], data['stage2'], "Stage 2: Final Output (Confidence-based Smoothing)", "d")

    # 공통 요소
    axes[3].set_xlabel("Time (seconds)", fontsize=12)
    
    # 컬러바 (데시벨 스케일)
    cbar = fig.colorbar(img, ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label('Amplitude (dB)', rotation=270, labelpad=15)

    # 4. 저장
    save_name = 'figure_real_experiment_result.png'
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"결과 이미지 저장 완료: {save_name}")
    plt.show()

if __name__ == "__main__":
    draw_real_spectrograms(FILE_PATHS)