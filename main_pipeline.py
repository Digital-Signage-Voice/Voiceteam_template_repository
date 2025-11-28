import sys
import os
import cv2
import numpy as np
import threading
import queue
import time
import sounddevice as sd
from faster_whisper import WhisperModel
import warnings
import noisereduce as nr
from PIL import Image, ImageDraw, ImageFont

warnings.filterwarnings("ignore")

# =========================================================
# [1단계] 경로 설정
# =========================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
video_path = os.path.join(src_path, 'video')
audio_module_path = os.path.join(current_dir, "audio-module", "src", "recognizer", "audio")

if video_path not in sys.path: sys.path.insert(0, video_path)
if src_path not in sys.path: sys.path.insert(1, src_path)
if audio_module_path not in sys.path: sys.path.append(audio_module_path)

from video.processor import VideoProcessor
from video.visualization.overlay import Overlay
from video.config import cfg

# =========================================================
# [2단계] 실시간 오디오 처리 설정 (Global 변수 및 함수)
# =========================================================
# 오디오 설정
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SIZE = 1024  # 한 번에 처리할 오디오 데이터 크기
TRANSCRIPTION_INTERVAL = 2.0  # STT 변환 주기 (초)

# 스레드 간 데이터 공유를 위한 큐와 변수
audio_queue = queue.Queue()
latest_stt_result = ""  # 화면에 표시할 최신 인식 텍스트

def audio_callback(indata, frames, time, status):
    """마이크에서 들어오는 오디오 데이터를 큐에 넣는 콜백 함수"""
    if status:
        print(status, file=sys.stderr)
    audio_queue.put(indata.copy())

def stt_worker(model):
    """백그라운드에서 오디오를 모아 STT를 수행하는 워커 스레드"""
    global latest_stt_result
    audio_buffer = np.array([], dtype=np.float32)
    
    print("🎙️ [Audio] STT 워커 스레드 시작됨")
    
    while True:
        try:
            # 1. 큐에 쌓인 오디오 데이터 수집
            while not audio_queue.empty():
                data = audio_queue.get()
                # 2차원 배열(프레임, 채널)을 1차원으로 평탄화하여 추가
                audio_buffer = np.concatenate((audio_buffer, data.flatten()))
            
            # [최적화] 버퍼가 너무 길면(예: 10초 이상) 최신 5초만 남기고 버림 (Backlog 방지)
            if len(audio_buffer) > SAMPLE_RATE * 10.0:
                print(f"⚠️ [STT] 버퍼 과부하! 오래된 오디오 삭제됨 ({len(audio_buffer)/SAMPLE_RATE:.1f}초 -> 5.0초)")
                keep_len = int(SAMPLE_RATE * 5.0)
                audio_buffer = audio_buffer[-keep_len:]

            # 2. 일정 시간 이상의 오디오가 모이면 STT 수행
            if len(audio_buffer) >= SAMPLE_RATE * TRANSCRIPTION_INTERVAL:
                # 분석할 구간만큼 잘라내기
                process_len = int(SAMPLE_RATE * TRANSCRIPTION_INTERVAL)
                chunk = audio_buffer[:process_len]
                audio_buffer = audio_buffer[process_len:] # 남은 부분은 유지 (오버랩 가능)
                
                # 노이즈 제거 (일단 비활성화)
                # chunk = nr.reduce_noise(y=chunk, sr=SAMPLE_RATE)
                
                print(f"🎤 [STT] 오디오 처리 중... (크기: {len(chunk)})")
                start_t = time.time()

                # Whisper STT 수행 (한국어) - faster-whisper
                # segments는 제너레이터이므로 리스트로 변환하여 텍스트 추출
                segments, info = model.transcribe(chunk, vad_filter=True, language="ko")
                text = " ".join([segment.text for segment in segments]).strip()
                
                if text:
                    latest_stt_result = text
                    print(f"🗣️ [인식됨]: {text}")
                else:
                    print("... (침묵 또는 인식 실패)")
                
                print(f"⏱️ [STT] 처리 시간: {time.time() - start_t:.2f}초")
            
            time.sleep(0.1) # CPU 점유율 조절
            
        except Exception as e:
            print(f"❌ STT Error: {e}")
            time.sleep(1)

def put_text_korean(img, text, position, font_size=20, color=(255, 255, 255)):
    """한글 텍스트를 이미지에 그리는 함수 (PIL 사용)"""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        # 윈도우 기본 폰트 (맑은 고딕)
        font = ImageFont.truetype("malgun.ttf", font_size)
    except:
        # 폰트가 없으면 기본 폰트 (한글 깨질 수 있음)
        font = ImageFont.load_default()
    
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# =========================================================
# [3단계] 메인 파이프라인 실행
# =========================================================
def run_realtime_pipeline():
    print(f"\n🚀 [Voice Team] 실시간 통합 파이프라인 시작 (Webcam + Mic)")

    # 1. Whisper 모델 로딩
    print("⏳ [Init] Whisper 모델 로딩 중... (faster-whisper base int8)")
    try:
        # faster-whisper 모델 로드 (CPU, INT8 양자화) - 가장 가벼운 base 모델 사용
        stt_model = WhisperModel("base", device="cpu", compute_type="int8")
        print("✅ 모델 로딩 완료!")
    except Exception as e:
        print(f"❌ Whisper 로딩 실패: {e}")
        return

    # 2. 오디오 스레드 시작
    # 데몬 스레드(daemon=True)로 설정하여 메인 프로그램 종료 시 같이 종료되게 함
    stt_thread = threading.Thread(target=stt_worker, args=(stt_model,), daemon=True)
    stt_thread.start()

    # 3. 마이크 녹음 시작 (SoundDevice)
    try:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE, 
            blocksize=BLOCK_SIZE, 
            device=None, # 기본 마이크 사용
            channels=CHANNELS, 
            callback=audio_callback
        )
        stream.start()
        print("✅ 마이크 입력 시작됨")
    except Exception as e:
        print(f"❌ 마이크 열기 실패: {e}")
        print("💡 팁: 사용 가능한 마이크가 없거나 권한이 없을 수 있습니다.")
        return

    # 4. 영상 프로세서 초기화
    # webcam 모드 사용 (path=None)
    # ML 모델 대신 규칙 기반 분류기 사용 (False)
    video_processor = VideoProcessor(source='webcam', path=None, visualize=False, use_ml=False)
    
    print("🎥 [Start] 영상 분석 시작... (종료: 화면 클릭 후 'q' 입력)")

    # VideoProcessor의 run() 대신 직접 루프를 돌려 STT 텍스트를 화면에 추가합니다.
    frame_id = 0
    while True:
        # (1) 영상 프레임 읽기
        ret, frame = video_processor.cap.read()
        if not ret:
            print("⚠️ 웹캠 신호 없음")
            break
        
        # 좌우 반전 (거울 모드)
        frame = cv2.flip(frame, 1)

        # (2) 영상 분석 (Lip Reading & Face Detection)
        result = video_processor.process_frame(frame_id, frame)
        
        # (3) 시각화 (기본 오버레이)
        frame_vis = Overlay.draw(frame.copy(), result)
        
        # (4) STT 결과 화면에 추가 (하단에 표시)
        h, w = frame_vis.shape[:2]
        # 텍스트 배경 박스
        cv2.rectangle(frame_vis, (0, h-60), (w, h), (0, 0, 0), -1)
        
        # 인식된 텍스트 (한글 출력을 위해 PIL 사용)
        stt_display = f"STT: {latest_stt_result}"
        frame_vis = put_text_korean(frame_vis, stt_display, (20, h-40), font_size=30, color=(255, 255, 255))

        # (5) 화면 출력
        cv2.imshow(cfg.window_name, frame_vis)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        frame_id += 1

    # 종료 처리
    stream.stop()
    stream.close()
    video_processor.cap.release()
    cv2.destroyAllWindows()
    print("👋 프로그램이 종료되었습니다.")

if __name__ == "__main__":
    run_realtime_pipeline()