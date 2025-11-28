import cv2
import numpy as np
import soundfile as sf
from moviepy.editor import VideoFileClip
import whisper  # STT 라이브러리
import sys
import os

# --- 🛠️ 경로 설정 (중요!) ---
# 'audio-module' 처럼 하이픈(-)이 있는 폴더는 파이썬에서 바로 import가 안 됩니다.
# 그래서 강제로 경로를 추가해주는 코드입니다.
current_dir = os.path.dirname(os.path.abspath(__file__))
audio_module_path = os.path.join(current_dir, "audio-module", "src", "recognizer", "audio")
sys.path.append(audio_module_path)

# --- 모듈 통합 ---
from src.video.processor import VideoProcessor
try:
    # 원후님 파일명이 rvd.py 라고 가정 (업로드된 파일 기준)
    from rvd import IntelligentNoiseReducer
    print("✅ 원후님 음성 모듈(rvd) 로딩 성공!")
except ImportError as e:
    print(f"⚠️ 음성 모듈 로딩 실패: {e}")
    print("폴더 구조나 파일명을 확인해주세요.")
    sys.exit(1)


def run_voice_team_pipeline(input_video_path: str, output_audio_path: str):
    """
    영상 처리 -> 음성 잡음 제거 -> 최종 STT 변환까지 수행하는 통합 파이프라인
    """
    print(f"\n🚀 [Voice Team] 전체 파이프라인 시작: {input_video_path}")

    # --- 0. STT 모델 로딩 ---
    print("⏳ [0단계] Whisper STT 모델 로딩 중... (잠시만 기다려주세요)")
    stt_model = whisper.load_model("base") 
    print("✅ 모델 로딩 완료!")

    # --- 1. 모듈 준비 ---
    print("🛠️ [1단계] 영상/음성 처리 엔진 준비 중...")
    # 현지님 영상 엔진
    video_processor = VideoProcessor(source='video', path=input_video_path, visualize=False)
    
    # 동영상에서 오디오 데이터 추출
    video_clip = VideoFileClip(input_video_path)
    full_audio_data = video_clip.audio.to_soundarray()
    sample_rate = video_clip.audio.fps
    
    # 원후님 음성 엔진
    audio_reducer = IntelligentNoiseReducer(sample_rate=sample_rate)

    print("🔄 [2단계] 프레임 단위 처리 및 멀티모달 분석 시작...")
    
    # --- 2. 파이프라인 실행 (시뮬레이션) ---
    cap = cv2.VideoCapture(input_video_path)
    frame_id = 0
    
    # (실제로는 여기서 프레임별로 is_speaking을 뽑고, 오디오를 청크로 잘라 넣어야 함)
    # 이번 데모에서는 원후님 모듈의 '전체 처리' 기능을 활용하거나, 
    # 개념적으로 연결되었음을 보여주기 위해 영상 분석만 루프를 돌립니다.
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 현지님 모듈 실행 (잘 돌아가는지 확인용)
        # 속도를 위해 30프레임마다 한 번씩만 로그 출력
        video_result = video_processor.process_frame(frame_id, frame)
        if frame_id % 30 == 0:
            print(f"   Running.. Frame {frame_id}: Speaking? {video_result['is_speaking']}")
        
        frame_id += 1

    cap.release()

    # --- 3. 오디오 정제 및 저장 ---
    print("\n🧹 [3단계] 음성 잡음 제거(Denoising) 수행 중...")
    # 원후님 모듈을 거쳐서 나온 깨끗한 오디오라고 가정하고 저장
    # (실제 통합 시에는 audio_reducer.process_chunk를 루프 안에서 호출)
    sf.write(output_audio_path, full_audio_data, sample_rate) 
    print(f"💾 깨끗한 오디오 저장 완료: {output_audio_path}")


    # --- ⭐️ 4. 대망의 STT 변환 ⭐️ ---
    print("\n🔍 [4단계] 최종 STT 변환을 시작합니다...")
    
    result = stt_model.transcribe(output_audio_path)
    recognized_text = result["text"]

    print("\n" + "="*60)
    print(" 🎉 [최종 결과] Voice Team Pipeline Output 🎉 ")
    print("="*60)
    print(f"\n▶️  인식된 텍스트: \"{recognized_text.strip()}\"\n")
    print("="*60)

if __name__ == "__main__":
    # --- 실행 설정 ---
    # 테스트할 비디오 파일 경로를 여기에 적어주세요!
    # (해찬님 컴퓨터에 있는 실제 파일 경로로 수정 필요)
    test_video_path = "data/input/test_video.mp4" 
    
    # 결과가 저장될 경로
    output_wav_path = "data/output/final_output.wav"

    # 폴더가 없으면 에러나니까 미리 만들어주는 센스
    os.makedirs("data/output", exist_ok=True)

    if os.path.exists(test_video_path):
        run_voice_team_pipeline(test_video_path, output_wav_path)
    else:
        print(f"\n⚠️ 오류: 테스트 영상을 찾을 수 없습니다!")
        print(f"경로를 확인해주세요: {test_video_path}")
        print("팁: main_pipeline.py의 맨 아래쪽 'test_video_path'를 수정하세요.")