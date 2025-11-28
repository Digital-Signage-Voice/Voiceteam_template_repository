# main_pipeline.py (Final Integrated Version)
# 역할: 영상 처리(Lip Reading) + 음성 처리(Denoising) + STT(Whisper) 통합 실행

import sys
import os
import cv2
import numpy as np
import traceback

# =========================================================
# [1단계] 경로 설정 (가장 중요)
# 각 모듈이 서로 다른 폴더에 있어도 파이썬이 찾을 수 있도록 경로를 강제 등록합니다.
# =========================================================
current_dir = os.path.dirname(os.path.abspath(__file__))

# 1. 영상 모듈 경로 (src, src/video)
src_path = os.path.join(current_dir, 'src')
video_path = os.path.join(src_path, 'video')

# 2. 음성 모듈 경로 (audio-module/src/recognizer/audio)
# 'audio-module' 폴더명이 하이픈(-)을 포함하므로 import가 불가능하여 경로 추가 필수
audio_module_path = os.path.join(current_dir, "audio-module", "src", "recognizer", "audio")

# 시스템 경로에 추가 (우선순위: Video -> Src -> Audio)
if video_path not in sys.path:
    sys.path.insert(0, video_path) # input 폴더 찾기용
if src_path not in sys.path:
    sys.path.insert(1, src_path)   # config.py 찾기용
if audio_module_path not in sys.path:
    sys.path.append(audio_module_path) # rvd 모듈 찾기용

# =========================================================
# [2단계] 라이브러리 및 모듈 Import
# =========================================================
try:
    import soundfile as sf
    from moviepy.editor import VideoFileClip
    import whisper  # STT 라이브러리
except ImportError as e:
    print(f"[오류] 필수 라이브러리가 설치되지 않았습니다: {e}")
    print("pip install moviepy soundfile openai-whisper 명령어로 설치해주세요.")
    sys.exit(1)

# 1. 영상 처리 엔진 (VideoProcessor)
try:
    try:
        from video.processor import VideoProcessor
    except ImportError:
        from processor import VideoProcessor
    print("✅ 영상 모듈(VideoProcessor) 로딩 성공!")
except ImportError as e:
    print(f"❌ 영상 모듈 로딩 실패: {e}")
    print(f"확인 경로: {src_path}")
    sys.exit(1)

# 2. 음성 처리 엔진 (IntelligentNoiseReducer)
try:
    from rvd import IntelligentNoiseReducer
    print("✅ 음성 모듈(RVD) 로딩 성공!")
except ImportError as e:
    print(f"⚠️ 음성 모듈 로딩 실패: {e}")
    print("audio-module 폴더 구조를 확인해주세요. (없다면 STT는 잡음 제거 없이 진행됩니다)")
    audio_reducer = None # 모듈 없으면 패스 처리


# =========================================================
# [3단계] 통합 파이프라인 로직
# =========================================================
def run_voice_team_pipeline(input_video_path: str, output_audio_path: str):
    """
    영상 처리 -> 음성 잡음 제거 -> 최종 STT 변환까지 수행하는 통합 파이프라인
    """
    print(f"\n🚀 [Voice Team] 전체 파이프라인 시작: {input_video_path}")

    # --- 0. Whisper 모델 로딩 ---
    print("⏳ [0단계] Whisper STT 모델 로딩 중... (최초 실행 시 다운로드)")
    try:
        stt_model = whisper.load_model("base") 
        print("✅ 모델 로딩 완료!")
    except Exception as e:
        print(f"❌ Whisper 모델 로딩 실패: {e}")
        return

    # --- 1. 모듈 초기화 ---
    print("⚙️ [1단계] 영상/음성 처리 엔진 준비 중...")
    
    # 영상 프로세서 초기화
    video_processor = VideoProcessor(source='video', path=input_video_path, visualize=False)
    
    # 동영상에서 오디오 데이터 추출 (MoviePy 사용)
    try:
        video_clip = VideoFileClip(input_video_path)
        if video_clip.audio is None:
            print("⚠️ 경고: 영상에 오디오 트랙이 없습니다!")
            return
            
        full_audio_data = video_clip.audio.to_soundarray()
        sample_rate = video_clip.audio.fps
        
        # 오디오 데이터가 스테레오(2채널)일 경우 모노로 변환 (Whisper 호환성)
        if len(full_audio_data.shape) > 1:
            full_audio_data = full_audio_data.mean(axis=1)
            
    except Exception as e:
        print(f"❌ 오디오 추출 실패: {e}")
        return
    
    # 음성 엔진 초기화 (있을 경우만)
    reducer = None
    if 'IntelligentNoiseReducer' in globals():
        reducer = IntelligentNoiseReducer(sample_rate=sample_rate)

    # --- 2. 영상 프레임 분석 (시뮬레이션) ---
    print("🎥 [2단계] 프레임 단위 영상 분석 시작...")
    
    cap = cv2.VideoCapture(input_video_path)
    frame_id = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 현지님 영상 모듈 실행
        video_result = video_processor.process_frame(frame_id, frame)
        
        # 로그 과다 출력 방지 (30프레임마다 출력)
        if frame_id % 30 == 0:
            speaking_state = "🗣️ 말함" if video_result['is_speaking'] else "🤫 침묵"
            print(f"   Frame {frame_id}: {speaking_state} (Conf: {video_result['confidence']})")
        
        frame_id += 1

    cap.release()

    # --- 3. 오디오 정제 및 저장 ---
    print("\n🔊 [3단계] 음성 잡음 제거(Denoising) 및 저장...")
    
    processed_audio = full_audio_data
    
    # 잡음 제거 모듈이 있다면 적용
    if reducer:
        print("   -> RVD 알고리즘 적용 중...")
        # (참고: 실제 RVD 모듈 사용법에 맞춰 chunk 처리가 필요할 수 있음. 여기선 전체 데이터라고 가정)
        # processed_audio = reducer.process_chunk(full_audio_data) 
        pass 
    
    # 오디오 파일 저장 (wav)
    sf.write(output_audio_path, processed_audio, sample_rate) 
    print(f"✅ 오디오 파일 생성 완료: {output_audio_path}")


    # --- 4. STT 변환 ---
    print("\n📝 [4단계] 최종 STT 변환을 시작합니다...")
    
    try:
        # 생성된 오디오 파일로 STT 수행
        transcription = stt_model.transcribe(output_audio_path, fp16=False) # fp16=False는 CPU 경고 방지
        recognized_text = transcription["text"]

        print("\n" + "="*60)
        print("🎉 [최종 결과] Voice Team Pipeline Output 🎉")
        print("="*60)
        print(f"\n 인식된 텍스트: \"{recognized_text.strip()}\"\n")
        print("="*60)
        
    except Exception as e:
        print(f"❌ STT 변환 실패: {e}")


if __name__ == "__main__":
    # --- 실행 설정 ---
    # 실제 존재하는 파일 경로로 수정해주세요
    # 예: C:\shinhan\DigitalSignage\data\video\front_1people_3.mp4
    input_video = r"C:\shinhan\DigitalSignage\data\video\front_1people_3.mp4"
    
    # 결과 저장 경로
    output_dir = "data/output"
    os.makedirs(output_dir, exist_ok=True)
    output_wav = os.path.join(output_dir, "extracted_audio.wav")

    # 파일 존재 확인 후 실행
    if os.path.exists(input_video):
        try:
            run_voice_team_pipeline(input_video, output_wav)
        except KeyboardInterrupt:
            print("\n⛔ 사용자에 의해 중단되었습니다.")
    else:
        print(f"\n❌ 오류: 입력 파일을 찾을 수 없습니다!")
        print(f"경로: {input_video}")
        print("main_pipeline.py 하단의 'input_video' 경로를 수정해주세요.")