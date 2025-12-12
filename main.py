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
from PIL import Image, ImageDraw, ImageFont
import textwrap

warnings.filterwarnings("ignore")

# =========================================================
# [1단계] 경로 및 모듈 설정
# =========================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
video_path = os.path.join(src_path, 'video')
# rvd.py가 있는 경로 설정 (구조에 맞게 수정)
audio_pkg_path = os.path.join(current_dir, "audio-module", "src", "recognizer", "audio")

if video_path not in sys.path: sys.path.insert(0, video_path)
if src_path not in sys.path: sys.path.insert(1, src_path)
if audio_pkg_path not in sys.path: sys.path.append(audio_pkg_path)

from video.processor import VideoProcessor
from video.visualization.overlay import Overlay
from video.config import cfg

# [NEW] 팀원이 작성한 잡음 제거 모듈 import
try:
    from rvd import FinalNoiseReducer, NoiseReducerConfig
except ImportError:
    # 경로 문제시 직접 파일명으로 임포트 시도 (환경에 따라 다를 수 있음)
    import rvd
    from rvd import FinalNoiseReducer, NoiseReducerConfig

# =========================================================
# [2단계] 전역 변수 및 큐 설정
# =========================================================
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SIZE = 1024 
TRANSCRIPTION_INTERVAL = 2.0 

# 1. 오디오 데이터 큐 (Raw Audio + Timestamp)
audio_queue = queue.Queue()
# 2. 비디오 분석 결과 큐 (VAD Info + Timestamp) - [NEW]
vad_queue = queue.Queue()

latest_stt_result = "" 

# 시각화용 상태 변수 (스레드 공유)
visual_context = {
    "person_detected": False,
    "is_speaking": False,
    "confidence": 0.0,
    "use_visual_gating": True 
}

def audio_callback(indata, frames, time_info, status):
    """마이크 입력 콜백: 데이터와 현재 시스템 시간을 함께 저장"""
    if status:
        print(status, file=sys.stderr)
    # [중요] 비디오와 동기화를 위해 현재 시간(time.time())을 함께 저장
    current_time = time.time()
    audio_queue.put((indata.copy(), current_time))

def stt_worker(model):
    """
    [통합 오디오 파이프라인]
    Raw Audio -> FinalNoiseReducer (Sync & Denoise) -> Clean Audio -> Whisper STT
    사람이 감지되지 않으면 오디오 데이터를 즉시 폐기하여 환각(Hallucination) 방지
    """
    global latest_stt_result
    
    # 1. 잡음 제거 모듈 초기화
    nr_config = NoiseReducerConfig(
        sample_rate=SAMPLE_RATE,
        low_latency_mode=True,
        vad_confidence_threshold=0.5, 
        stage1_prop_decrease=0.9,
        bypass_mode=False
    )
    reducer = FinalNoiseReducer(nr_config)
    
    stt_accum_buffer = np.array([], dtype=np.float32)
    
    # [추가] 마지막으로 발화가 감지된 시간 (초기값: 현재시간)
    last_speech_time = time.time()
    
    # [추가] 여유 시간 설정 (초)
    # 1.5초 동안은 입을 다물어도 "말하는 중"으로 간주하고 기다림
    PAUSE_THRESHOLD = 1.5
    
    print("🎙️ [Audio] 통합 노이즈 제거 및 STT 워커 시작")
    
    while True:
        try:
            current_time = time.time()

            # [0] 발화 타이머 갱신 (핵심 로직)
            if visual_context["person_detected"] and visual_context["is_speaking"]:
                last_speech_time = current_time
            
            # 마지막 발화로부터 얼마나 지났는지 계산
            time_since_speech = current_time - last_speech_time
            
            # [1] 데이터 주입 (Queue -> Reducer)
            while not vad_queue.empty():
                vad_data = vad_queue.get()
                if not visual_context["use_visual_gating"]:
                     reducer.add_vad_result(True, 1.0, vad_data['timestamp'])
                else:
                    reducer.add_vad_result(
                        vad_data['is_speaking'], 
                        vad_data['confidence'], 
                        vad_data['timestamp']
                    )

            while not audio_queue.empty():
                raw_audio, timestamp = audio_queue.get()
                flat_audio = raw_audio.flatten().astype(np.float32)
                reducer.add_audio_chunk(flat_audio, timestamp)

            # [2] Visual Gating
            # 차단해야 하는 경우:
            # 1. 사람이 아예 없을 때 (즉시 차단)
            # 2. 사람은 있지만 입을 다문지 1.5초가 지났을 때 (지연 차단)
            
            should_block_stt = False
            block_reason = ""
            
            if visual_context["use_visual_gating"]:
                if not visual_context["person_detected"]:
                    should_block_stt = True
                    block_reason = "사람 없음"
                elif time_since_speech > PAUSE_THRESHOLD:
                    should_block_stt = True
                    block_reason = f"침묵 {time_since_speech:.1f}초 경과"

            if should_block_stt:
                # 데이터 버리기
                _ = reducer.get_processed_chunk() 
                
                # 버퍼가 차있었다면, 지워지기 전에 로그 출력 (아까운 데이터 확인)
                if len(stt_accum_buffer) > 0:
                    # print(f"🧹 [Reset] 버퍼 초기화됨 ({block_reason})")
                    pass
                    
                stt_accum_buffer = np.array([], dtype=np.float32)
                time.sleep(0.05)
                continue

            # [3] 오디오 회수
            while True:
                clean_chunk = reducer.get_processed_chunk()
                if clean_chunk is None: break
                stt_accum_buffer = np.concatenate((stt_accum_buffer, clean_chunk))

            # 버퍼 길이 제한
            if len(stt_accum_buffer) > SAMPLE_RATE * 10.0:
                stt_accum_buffer = stt_accum_buffer[-int(SAMPLE_RATE*5.0):]

            # [4] STT 추론
            if len(stt_accum_buffer) >= SAMPLE_RATE * TRANSCRIPTION_INTERVAL:
                process_len = int(SAMPLE_RATE * TRANSCRIPTION_INTERVAL)
                chunk_to_transcribe = stt_accum_buffer[:process_len]
                stt_accum_buffer = stt_accum_buffer[process_len:] 

                # RMS 기준 대폭 완화 (0.005 -> 0.001)
                rms = np.sqrt(np.mean(chunk_to_transcribe**2))
                
                # 디버깅: RMS 값이 너무 작아서 무시되는지 확인
                # print(f"📊 [Check] RMS: {rms:.5f}") 

                if rms < 0.001: 
                    continue

                # Whisper 파라미터
                segments, info = model.transcribe(
                    chunk_to_transcribe, 
                    vad_filter=True, 
                    language="ko",
                    temperature=0.0,            # 창의성 제거
                    condition_on_previous_text=False,
                    no_speech_threshold=0.6     # 말 아님 확률 높으면 무시
                )
                
                text = " ".join([segment.text for segment in segments]).strip()
                
                # 환각 필터링
                hallucination_filters = ["MBC", "뉴스", "시청해", "감사합니다", "구독", "좋아요"]
                if any(h in text for h in hallucination_filters) and len(text) < 15:
                    continue

                if text:
                    latest_stt_result = text
                    print(f"🗣️ [STT]: {text}")

            time.sleep(0.01)

        except Exception as e:
            print(f"❌ Worker Error: {e}")
            time.sleep(1)

# =========================================================
# [3단계] 텍스트 그리기 유틸리티
# =========================================================
def put_text_korean(img, text, position, font_size=20, color=(255, 255, 255), max_width=None):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("malgun.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    x, y = position
    lines = textwrap.wrap(text, width=int(max_width / (font_size * 0.7))) if max_width else [text]

    for line in lines:
        draw.text((x, y), line, font=font, fill=color)
        y += int(font_size * 1.5)

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# =========================================================
# [4단계] 메인 실행 함수
# =========================================================
def run_realtime_pipeline():
    print(f"\n🚀 [Voice Team] 통합 파이프라인 (Video-VAD + Denoise + STT) 시작")

    # 1. 모델 로드
    print("⏳ Whisper 모델 로딩 중...")
    try:
        stt_model = WhisperModel("small", device="cpu", compute_type="int8")
    except Exception as e:
        print(f"❌ Whisper 로딩 실패: {e}")
        return

    # 2. 스레드 시작
    stt_thread = threading.Thread(target=stt_worker, args=(stt_model,), daemon=True)
    stt_thread.start()

    # 3. 마이크 스트림 시작
    try:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, 
            channels=CHANNELS, callback=audio_callback
        )
        stream.start()
    except Exception as e:
        print(f"❌ 마이크 실패: {e}")
        return

    # 4. 영상 처리 루프
    video_processor = VideoProcessor(source='webcam', path=None, visualize=False, use_ml=False)
    print("🎥 영상 분석 시작... (종료: q, 모드전환: t)")

    frame_id = 0
    while True:
        # (1) 프레임 획득
        ret, frame = video_processor.cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        
        # [중요] 오디오와 동기화를 위한 타임스탬프 생성 (시스템 시간 기준)
        current_ts = time.time()

        # (2) 영상 분석 (processor.py 로직)
        # processor 내부 timestamp 대신 current_ts를 사용하도록 결과 딕셔너리 수정 필요할 수 있음
        result = video_processor.process_frame(frame_id, frame)
        
        # 타임스탬프 덮어쓰기 (Audio Callback의 time.time()과 동기화)
        result['timestamp'] = current_ts 
        
        # (3) 결과 공유
        # A. UI 표시용 (최신 상태만 필요)
        visual_context["person_detected"] = result["person_detected"]
        visual_context["is_speaking"] = result["is_speaking"]
        visual_context["confidence"] = result["confidence"]
        
        # B. 잡음 제거용 (모든 프레임 히스토리 필요 -> 큐 전송)
        #    여기에 1단계에서 산출된 roi, confidence, is_speaking 정보가 모두 담겨 오디오 스레드로 이동
        vad_queue.put(result)

        # (4) 시각화
        frame_vis = Overlay.draw(frame.copy(), result)
        h, w = frame_vis.shape[:2]

        # Blackout 처리 (오디오 모드 제외)
        if visual_context["use_visual_gating"] and not visual_context["person_detected"]:
             # 사람이 없으면 화면 어둡게 (프라이버시/절전) - 필요시 로직 변경
             pass 

        # 하단 자막 영역
        cv2.rectangle(frame_vis, (0, h-120), (w, h), (0, 0, 0), -1)
        
        # 상태 텍스트 결정
        if not visual_context["use_visual_gating"]:
            status_msg = "Audio Mode Only"
            status_color = (0, 255, 255) # 노랑
        elif result["is_speaking"]:
            status_msg = f"Speaking (Conf: {result['confidence']:.2f})"
            status_color = (0, 255, 0)   # 초록
        elif result["person_detected"]:
            status_msg = "Silent"
            status_color = (200, 200, 200) # 회색
        else:
            status_msg = "Searching..."
            status_color = (0, 0, 255)   # 빨강

        cv2.putText(frame_vis, status_msg, (20, h-90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        frame_vis = put_text_korean(frame_vis, latest_stt_result, (20, h-60), font_size=28, max_width=w-40)

        cv2.imshow(cfg.window_name, frame_vis)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('t'):
            visual_context["use_visual_gating"] = not visual_context["use_visual_gating"]
            print(f"🔄 모드 전환: {'A-V Fusion' if visual_context['use_visual_gating'] else 'Audio Only'}")
            
        frame_id += 1

    stream.stop()
    stream.close()
    video_processor.cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_realtime_pipeline()