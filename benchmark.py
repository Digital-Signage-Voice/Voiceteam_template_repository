import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import cv2
import warnings
from moviepy.editor import VideoFileClip

# [NEW] STT 모듈 임포트
try:
    from faster_whisper import WhisperModel
except ImportError:
    print("❌ faster_whisper가 설치되지 않았습니다. 'pip install faster-whisper'를 실행하세요.")
    sys.exit(1)

warnings.filterwarnings("ignore")

# =========================================================
# [1단계] 경로 및 모듈 설정
# =========================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
video_path = os.path.join(src_path, 'video')
audio_pkg_path = os.path.join(current_dir, "audio-module", "src", "recognizer", "audio")

if video_path not in sys.path: sys.path.insert(0, video_path)
if src_path not in sys.path: sys.path.insert(1, src_path)
if audio_pkg_path not in sys.path: sys.path.append(audio_pkg_path)

try:
    from video.processor import VideoProcessor
    from rvd import FinalNoiseReducer, NoiseReducerConfig
    print("✅ [Import] VideoProcessor & RVD 모듈 로드 성공")
except ImportError as e:
    print(f"❌ [Error] 모듈 로드 실패: {e}")
    sys.exit(1)

# =========================================================
# [2단계] 설정
# =========================================================
VIDEO_PATH = "C:/shinhan/DigitalSignage/data/video/IMG_4636.MOV" # 동영상 경로
OUTPUT_DIR = os.path.join(current_dir, "benchmark_outputs")
TEMP_WAV_PATH = os.path.join(OUTPUT_DIR, "temp_extracted.wav")
TRANSCRIPT_FILE = os.path.join(OUTPUT_DIR, "benchmark_transcriptions.txt")

SAMPLE_RATE = 16000
CHUNK_SIZE = 4096 

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# =========================================================
# [Helper] 오디오 추출 함수
# =========================================================
def load_audio_from_video(video_path, sr=16000, duration=None):
    print(f"🔊 [Audio] 비디오에서 오디오 추출 중... ({os.path.basename(video_path)})")
    try:
        video_clip = VideoFileClip(video_path)
        audio_clip = video_clip.audio
        
        if duration is not None:
            end_time = min(duration, video_clip.duration)
            audio_clip = audio_clip.subclip(0, end_time)
            
        audio_clip.write_audiofile(TEMP_WAV_PATH, fps=sr, nbytes=2, codec='pcm_s16le', logger=None)
        audio_clip.close()
        video_clip.close()
        
        y, _ = librosa.load(TEMP_WAV_PATH, sr=sr)
        
        if os.path.exists(TEMP_WAV_PATH):
            os.remove(TEMP_WAV_PATH)
        return y
    except Exception as e:
        print(f"❌ [Error] 오디오 추출 실패: {e}")
        return np.array([])

# =========================================================
# [3단계] Visual VAD 추출기
# =========================================================
def extract_visual_vad(video_path):
    print("🎥 [Video] Visual VAD 정보 추출 중...")
    
    vp = VideoProcessor(source='video', path=video_path, visualize=False, use_ml=True)
    
    try:
        fps = vp.cap.fps
    except AttributeError:
        fps = 30.0
    
    print(f"   -> Detected FPS: {fps}")

    vad_history = [] 
    frame_id = 0
    start_time = time.time()
    
    while True:
        ret, frame = vp.cap.read()
        if not ret: break
        
        result = vp.process_frame(frame_id, frame)
        ts = frame_id / fps
        
        vad_history.append({
            'timestamp': ts,
            'is_speaking': result.get('is_speaking', False),
            'confidence': result.get('confidence', 0.0)
        })
        
        frame_id += 1
        if frame_id % 100 == 0:
            print(f"   -> {frame_id} frames processed...", end='\r')

    vp.cap.release()
    print(f"\n✅ [Video] 분석 완료: {len(vad_history)} frames")
    return vad_history, fps

# =========================================================
# [4단계] Audio Processing Pipeline (Safe Mode)
# =========================================================
def run_audio_pipeline(audio_data, config, vad_history, mode='standard'):
    reducer = FinalNoiseReducer(config)
    processed_chunks = []
    
    # 패딩: 전체 길이를 CHUNK_SIZE 배수로 맞춤
    original_len = len(audio_data)
    remainder = original_len % CHUNK_SIZE
    if remainder != 0:
        pad_len = CHUNK_SIZE - remainder
        padded_audio = np.concatenate((audio_data, np.zeros(pad_len, dtype=audio_data.dtype)))
    else:
        padded_audio = audio_data
        
    total_samples = len(padded_audio)
    idx = 0
    start_time = time.time()
    
    vad_idx = 0
    max_vad_idx = len(vad_history) - 1
    
    try:
        while idx < total_samples:
            end = idx + CHUNK_SIZE 
            chunk = padded_audio[idx:end]
            current_ts = idx / SAMPLE_RATE
            
            # Sync Logic
            if mode == 'proposed' and vad_history:
                while vad_idx < max_vad_idx and vad_history[vad_idx+1]['timestamp'] < current_ts:
                    vad_idx += 1
                v_data = vad_history[vad_idx]
                reducer.add_vad_result(v_data['is_speaking'], v_data['confidence'], current_ts)
            else:
                reducer.add_vad_result(True, 1.0, current_ts)

            # 오디오 처리
            reducer.add_audio_chunk(chunk, current_ts)
            
            while True:
                out = reducer.get_processed_chunk()
                if out is None: break
                processed_chunks.append(out)
                
            idx += CHUNK_SIZE
            
        # Flush
        remain = reducer.flush()
        if len(remain) > 0:
            processed_chunks.append(remain)

    except Exception as e:
        print(f"\n⚠️ [Warning] 잡음 제거 중 에러 발생 (Skipping): {e}")
        # 에러 발생 시 원본(패딩된 버전)을 그대로 반환해서 STT라도 되게 함
        return padded_audio[:original_len], 0.0
        
    process_duration = time.time() - start_time
    
    if processed_chunks:
        result_audio = np.concatenate(processed_chunks)
    else:
        result_audio = np.zeros_like(padded_audio)

    # 길이 복원
    if len(result_audio) > original_len:
        result_audio = result_audio[:original_len]
    elif len(result_audio) < original_len:
        result_audio = np.pad(result_audio, (0, original_len - len(result_audio)))
        
    return result_audio, process_duration

# =========================================================
# [5단계] 지표 계산
# =========================================================
def calculate_metrics_with_vad(original, processed, proc_time, vad_history, sr):
    duration = len(original) / sr
    rtf = proc_time / duration if duration > 0 else 0
    
    hop = 512
    orig_rms = np.array([np.mean(original[i:i+hop]**2) for i in range(0, len(original), hop)])
    proc_rms = np.array([np.mean(processed[i:i+hop]**2) for i in range(0, len(processed), hop)])
    
    orig_db = 10 * np.log10(np.maximum(orig_rms, 1e-10))
    proc_db = 10 * np.log10(np.maximum(proc_rms, 1e-10))
    
    is_speaking_mask = np.zeros(len(orig_db), dtype=bool)
    for v in vad_history:
        idx = int(v['timestamp'] * sr / hop)
        if idx < len(is_speaking_mask):
            is_speaking_mask[idx] = v['is_speaking']
            
    if np.any(is_speaking_mask):
        speech_loss = np.mean(orig_db[is_speaking_mask]) - np.mean(proc_db[is_speaking_mask])
    else:
        speech_loss = 0.0 
    
    if np.any(~is_speaking_mask):
        noise_reduction = np.mean(orig_db[~is_speaking_mask]) - np.mean(proc_db[~is_speaking_mask])
    else:
        noise_reduction = 0.0

    return {"RTF": rtf, "Noise_Reduction_dB": noise_reduction, "Speech_Loss_dB": speech_loss}

# =========================================================
# [New] STT 수행 함수
# =========================================================
def perform_stt(model, audio_data, case_name):
    print(f"📝 [STT] {case_name} 변환 중...")
    start = time.time()
    try:
        segments, info = model.transcribe(audio_data, beam_size=5, language="ko", vad_filter=True)
        text = " ".join([segment.text for segment in segments]).strip()
        elapsed = time.time() - start
        print(f"   -> 완료 ({elapsed:.2f}s): {text[:30]}...")
        return text
    except Exception as e:
        print(f"   -> STT 실패: {e}")
        return "(STT Error)"

# =========================================================
# [6단계] Main Benchmark
# =========================================================
def run_benchmark():
    print(f"\n🚀 [Benchmark] 성능 분석 및 STT 통합 테스트")
    
    # 0. Whisper 모델 로드
    print("⏳ Whisper 모델 로딩 중... (cpu/int8)")
    try:
        model = WhisperModel("base", device="cpu", compute_type="int8")
    except Exception as e:
        print(f"❌ Whisper 로드 실패: {e}")
        return

    # 1. Load Audio
    raw_audio = load_audio_from_video(VIDEO_PATH, sr=SAMPLE_RATE, duration=60)
    if len(raw_audio) == 0: return
    
    # 2. Extract VAD
    vad_history, fps = extract_visual_vad(VIDEO_PATH)
    
    results = {}      # 성능 지표 저장
    transcripts = {}  # STT 텍스트 저장

    # --- Case 1: Raw ---
    print("\n1️⃣ [Case 1] Raw Audio")
    sf.write(os.path.join(OUTPUT_DIR, "1_raw.wav"), raw_audio, SAMPLE_RATE)
    
    transcripts['Raw'] = perform_stt(model, raw_audio, "Raw Audio")
    results['Raw'] = {"RTF": 0.0, "Noise_Reduction_dB": 0.0, "Speech_Loss_dB": 0.0}

    # --- Case 2: Standard ---
    print("\n2️⃣ [Case 2] Standard NR")
    cfg_std = NoiseReducerConfig(sample_rate=SAMPLE_RATE, stationary=True, non_speech_gain=1.0, vad_confidence_threshold=0.0)
    
    audio_std, time_std = run_audio_pipeline(raw_audio, cfg_std, vad_history, mode='standard')
    
    sf.write(os.path.join(OUTPUT_DIR, "2_standard.wav"), audio_std, SAMPLE_RATE)
    transcripts['Standard'] = perform_stt(model, audio_std, "Standard NR")
    
    metrics_std = calculate_metrics_with_vad(raw_audio, audio_std, time_std, vad_history, SAMPLE_RATE)
    results['Standard'] = metrics_std
    print(f"   👉 RTF: {metrics_std['RTF']:.4f} | NR: {metrics_std['Noise_Reduction_dB']:.2f}dB")

    # --- Case 3: Proposed ---
    print("\n3️⃣ [Case 3] Proposed NR (Visual VAD)")
    cfg_pro = NoiseReducerConfig(sample_rate=SAMPLE_RATE, stationary=False, vad_confidence_threshold=0.5, non_speech_gain=0.1, low_latency_mode=True)
    
    audio_pro, time_pro = run_audio_pipeline(raw_audio, cfg_pro, vad_history, mode='proposed')
    
    sf.write(os.path.join(OUTPUT_DIR, "3_proposed.wav"), audio_pro, SAMPLE_RATE)
    transcripts['Proposed'] = perform_stt(model, audio_pro, "Proposed NR")
    
    metrics_pro = calculate_metrics_with_vad(raw_audio, audio_pro, time_pro, vad_history, SAMPLE_RATE)
    results['Proposed'] = metrics_pro
    print(f"   👉 RTF: {metrics_pro['RTF']:.4f} | NR: {metrics_pro['Noise_Reduction_dB']:.2f}dB")

    # ------------------------------------
    # STT 결과 파일 저장
    # ------------------------------------
    print(f"\n💾 STT 결과 저장 중: {TRANSCRIPT_FILE}")
    with open(TRANSCRIPT_FILE, "w", encoding="utf-8") as f:
        f.write("=== Benchmark Transcription Results ===\n")
        f.write(f"Target Video: {os.path.basename(VIDEO_PATH)}\n\n")
        
        f.write("[1. Raw Audio]\n")
        f.write(transcripts['Raw'] + "\n\n")
        
        f.write("[2. Standard NR]\n")
        f.write(transcripts['Standard'] + "\n\n")
        
        f.write("[3. Proposed NR]\n")
        f.write(transcripts['Proposed'] + "\n")
        
    # ------------------------------------
    # 시각화
    # ------------------------------------
    print("📊 그래프 생성 중...")
    labels = ['Standard', 'Proposed']
    nr = [results['Standard']['Noise_Reduction_dB'], results['Proposed']['Noise_Reduction_dB']]
    sl = [results['Standard']['Speech_Loss_dB'], results['Proposed']['Speech_Loss_dB']]
    rtf = [results['Standard']['RTF'], results['Proposed']['RTF']]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(x - width/2, nr, width, label='Noise Reduction (dB) ↑', color='skyblue')
    ax1.bar(x + width/2, sl, width, label='Speech Loss (dB) ↓', color='salmon')
    ax1.set_ylabel('dB')
    ax1.set_title('RVD Benchmark (with STT)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend(loc='upper left')
    
    ax2 = ax1.twinx()
    ax2.plot(x, rtf, color='green', marker='o', label='RTF ↓')
    ax2.set_ylabel('RTF')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "benchmark_chart.png"))
    plt.show()
    print("✅ 모든 작업 완료.")

if __name__ == "__main__":
    run_benchmark()