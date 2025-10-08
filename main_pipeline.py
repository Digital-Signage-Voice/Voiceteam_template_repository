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
    
    # 동영상 파일을 한 프레임씩 읽기 위해 OpenCV를 사용합니다.
    cap = cv2.VideoCapture(input_video_path)
    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ✨ 핵심 통합 지점 ✨
        # 현재 프레임을 현지님의 엔진에 넣고, 분석 결과를 딕셔너리로 받습니다.
        result_dict = video_processor.process_frame(frame_id, frame)
        
        is_speaking = result_dict['is_speaking']
        timestamp = result_dict['timestamp']

        print(f"  - Frame #{frame_id}: 현재 시간 {timestamp:.2f}초, 발화 여부: {is_speaking}")

        # 만약 '발화 중' 이라면, 해당 시간대의 오디오에 원후님의 모듈을 적용
        # (이 부분은 추후 원후님 모듈이 실시간 청크 단위를 지원하면 더 정교화될 예정)
        # 지금은 간단히 'speech_segments' 정보를 활용
        if result_dict['speech_segments']:
             for segment in result_dict['speech_segments']:
                start_time = segment.get('start')
                # 'end'는 아직 없으므로, 지금은 이 로직을 단순화하여 적용
                # 이 부분은 앞으로 원후님과 함께 더 발전시켜야 합니다.

        frame_id += 1
    
    cap.release()
    print("\n✅ 영상의 모든 프레임 처리가 완료되었습니다.")

    # (임시) 현재는 오프라인 방식이므로, 분석 결과를 바탕으로 후처리하는 로직이 필요
    # 이 부분은 추후 실시간 구조로 변경되면서 수정될 것입니다.
    # 지금은 원후님 모듈을 호출하는 부분을 비워두고, 파이프라인이 연결되는 것만 확인합니다.

    # --- 4단계: 최종 결과물 저장 ---
    print("\n💾 4단계: 최종 오디오를 파일로 저장합니다...")
    # sf.write(output_audio_path, final_audio_data, sample_rate) # 아직 실제 처리가 없으므로 주석 처리
    print(f"🎉 성공! 파이프라인 실행이 완료되었습니다. (경로: '{output_audio_path}')")


# --- 이 파일을 직접 실행했을 때만 아래 코드가 동작합니다 ---
if __name__ == "__main__":
    test_video = "data/input/your_test_video.mp4"  # 실제 테스트할 영상 파일 경로를 넣어주세요.
    output_wav = "data/output/final_clean_audio.wav"

    run_voice_team_pipeline(test_video, output_wav)