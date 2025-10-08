# main_pipeline.py (v2 - Updated by Haechan)
# 역할: 전체 파이프라인을 총괄하며, 각 팀원의 '완성된 모듈'을 가져와 조립하는 지휘자

import cv2
import numpy as np
from moviepy.editor import VideoFileClip
import soundfile as sf

# --- 모듈 통합 지점 ---
# 현지님의 '영상 처리 엔진' 클래스를 불러옵니다.
# (이제 video_module.py 대신, 현지님이 만든 processor.py를 사용합니다)
from src.video.processor import VideoProcessor

# 원후님의 '음성 처리 엔진' 함수를 불러옵니다. (기존과 동일)
from src.audio.audio_module import denoise_audio


def run_voice_team_pipeline(input_video_path: str, output_audio_path: str):
    """
    우리 Voice팀의 전체 파이프라인을 실행합니다.
    """
    print("🚀 Voice팀 전체 파이프라인 v2를 시작합니다!")

    # --- 1단계: 모듈 준비 ---
    # 현지님의 영상 처리 엔진을 생성합니다. (오프라인 영상 파일 모드)
    print("🛠️  1단계: 현지님의 영상 처리 엔진을 준비합니다...")
    # source='video'로 설정하여 파일 기반으로 작동하게 합니다.
    video_processor = VideoProcessor(source='video', path=input_video_path, visualize=False)

    # --- 2단계: 재료 준비 ---
    print("🔪 2단계: 동영상에서 오디오를 추출합니다...")
    video_clip = VideoFileClip(input_video_path)
    full_audio_data = video_clip.audio.to_soundarray()
    sample_rate = video_clip.audio.fps
    final_audio_data = full_audio_data.copy()

    # --- 3단계: 파이프라인 실행 (프레임 단위 처리) ---
print("\n🔄 3단계: 영상의 각 프레임을 순서대로 처리하며 파이프라인을 실행합니다...")

cap = cv2.VideoCapture(input_video_path)
frame_id = 0
fps = cap.get(cv2.CAP_PROP_FPS)  # 영상의 초당 프레임 수
frame_duration = 1 / fps        # 한 프레임이 차지하는 시간 (초)

processed_audio_segments = []  # 처리된 오디오 조각들을 모아둘 리스트

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 1. 현지님 모듈로 영상 프레임 분석
    result_dict = video_processor.process_frame(frame_id, frame)
    is_speaking = result_dict['is_speaking']
    timestamp = result_dict['timestamp']

    print(f"  - Frame #{frame_id}: 현재 시간 {timestamp:.2f}초, 발화 여부: {is_speaking}")

    # 2. 타임스탬프를 이용해 현재 프레임에 해당하는 오디오 조각 추출
    start_sample = int(timestamp * sample_rate)
    end_sample = int((timestamp + frame_duration) * sample_rate)
    audio_chunk = full_audio_data[start_sample:end_sample]

    # 3. ✨ 핵심 통합 로직 ✨
    if is_speaking:
        # '발화 중'이면 원후님 모듈로 오디오 처리 (예: 노이즈 제거)
        processed_chunk = denoise_audio(audio_chunk, sample_rate)
        processed_audio_segments.append(processed_chunk)
    else:
        # '비발화 중'이면 소리를 0으로 만들어 Mute 처리
        silence_chunk = np.zeros_like(audio_chunk)
        processed_audio_segments.append(silence_chunk)

    frame_id += 1

cap.release()
print("\n✅ 영상의 모든 프레임 처리가 완료되었습니다.")

# --- 4단계: 후처리 및 결과물 저장 ---
print("\n💾 4단계: 처리된 오디오 조각들을 하나로 합쳐 파일로 저장합니다...")

# 모든 오디오 조각들을 하나로 합칩니다.
final_audio_data = np.concatenate(processed_audio_segments)

sf.write(output_audio_path, final_audio_data, sample_rate)
print(f"🎉 성공! 파이프라인 실행이 완료되었습니다. (경로: '{output_audio_path}')")
