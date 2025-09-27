# 예시 코드입니다~~ 약간 이런 방향으로 정의를 해보았습니다!!

# audio_module.py
# 역할: 오디오 데이터를 입력받아, 잡음을 제거하는 기술을 적용하는 모듈
import numpy as np

# 원후님은 이 함수 안에 뭐 예를들어 Spectral Subtraction 같은 잡음 제거 기술을 구현하면
def denoise_audio(audio_chunk: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    오디오 데이터(Numpy 배열)를 입력받아, 잡음을 제거한 뒤 결과 데이터를 반환합니다.

    Args:
        audio_chunk (np.ndarray): 잡음을 제거할 오디오 데이터 조각
        sample_rate (int): 오디오의 샘플링 레이트 (예: 44100)

    Returns:
        np.ndarray: 잡음이 제거된 오디오 데이터 조각
    """
    print("🔊 오디오 잡음 제거 기술을 적용합니다...")

    # --- 원후님이 실제 기술을 구현할 부분 ---
    # (지금은 테스트를 위해 소리를 살짝 줄이는 가짜 효과를 적용합니다)
    denoised_chunk = audio_chunk * 0.5 # 실제로는 복잡한 알고리즘이 들어감
    # ------------------------------------

    print("✅ 잡음 제거 완료!")
    return denoised_chunk