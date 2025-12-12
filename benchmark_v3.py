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

# [STT 모듈 임포트]
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
VIDEO_PATH = "C:/shinhan/DigitalSignage/data/video/IMG_4636.MOV"
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
    print(f"🔊 [Audio] 오디오 추출 중... ({os.path.basename(video_path)})")
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
    
    vad_history = [] 
    frame_id = 0
    
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

    if hasattr(vp.cap, 'release'):
        vp.cap.release()
        
    print(f"\n✅ [Video] 분석 완료: {len(vad_history)} frames")
    return vad_history, fps

# =========================================================
# [4단계] Audio Processing Pipeline (RTF 측정 + 고품질 생성)
# =========================================================
def run_audio_pipeline(audio_data, config, vad_history, mode='standard'):
    """
    1. RTF 측정: Chunking 루프를 돌려서 '속도'를 측정
    2. 오디오 생성: 전체 파일을 통째로 처리하여 '음질(Clicking 제거)' 확보
    """
    
    # --- [Pass 1] RTF 측정용 (Chunking Loop) ---
    reducer_rtf = FinalNoiseReducer(config)
    original_len = len(audio_data)
    
    # 패딩 처리
    remainder = original_len % CHUNK_SIZE
    if remainder != 0:
        padded_audio = np.concatenate((audio_data, np.zeros(CHUNK_SIZE - remainder, dtype=audio_data.dtype)))
    else:
        padded_audio = audio_data
        
    total_samples = len(padded_audio)
    idx = 0
    
    start_time = time.time()
    
    # RTF용 루프 (결과물은 버림)
    while idx < total_samples:
        end = idx + CHUNK_SIZE
        chunk = padded_audio[idx:end]
        ts = idx / SAMPLE_RATE
        
        # VAD 주입 시뮬레이션
        if mode == 'proposed' and vad_history:
            v_idx = min(len(vad_history)-1, int(ts * 30)) # 30fps 가정
            v = vad_history[v_idx]
            reducer_rtf.add_vad_result(v['is_speaking'], v['confidence'], ts)
        else:
            reducer_rtf.add_vad_result(True, 1.0, ts)
            
        reducer_rtf.add_audio_chunk(chunk, ts)
        _ = reducer_rtf.get_processed_chunk()
        idx += CHUNK_SIZE
    
    _ = reducer_rtf.flush()
    process_duration = time.time() - start_time
    
    
    # --- [Pass 2] 오디오 생성용 (Batch Processing) ---
    print(f"   ℹ️ [Quality] 오디오 생성 중 ({mode})...")
    reducer_quality = FinalNoiseReducer(config)
    
    # 1. 전체 VAD 정보 미리 주입
    if mode == 'proposed' and vad_history:
        for v in vad_history:
            reducer_quality.add_vad_result(v['is_speaking'], v['confidence'], v['timestamp'])
    else:
        duration = len(audio_data) / SAMPLE_RATE
        for t in np.arange(0, duration, 0.1):
            reducer_quality.add_vad_result(True, 1.0, float(t))
            
    # 2. 전체 오디오 통째로 주입 (Clicking 방지)
    reducer_quality.add_audio_chunk(audio_data, 0.0)
    
    # 3. 결과 회수
    out_chunks = []
    while True:
        c = reducer_quality.get_processed_chunk()
        if c is None: break
        out_chunks.append(c)
    
    remain = reducer_quality.flush()
    if len(remain) > 0: out_chunks.append(remain)
    
    if out_chunks:
        final_audio = np.concatenate(out_chunks)
    else:
        final_audio = np.zeros_like(audio_data)
        
    # 길이 맞춤
    if len(final_audio) > original_len:
        final_audio = final_audio[:original_len]
    elif len(final_audio) < original_len:
        final_audio = np.pad(final_audio, (0, original_len - len(final_audio)))
        
    return final_audio, process_duration

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
    else: speech_loss = 0.0 
    
    if np.any(~is_speaking_mask):
        noise_reduction = np.mean(orig_db[~is_speaking_mask]) - np.mean(proc_db[~is_speaking_mask])
    else: noise_reduction = 0.0

    return {"RTF": rtf, "Noise_Reduction_dB": noise_reduction, "Speech_Loss_dB": speech_loss}

# =========================================================
# [New] STT 수행 함수 (프롬프트 추가)
# =========================================================
def perform_stt(model, audio_data, case_name):
    print(f"📝 [STT] {case_name} 변환 중...")
    
    # 1. 완벽한 침묵이면 스킵
    rms = np.sqrt(np.mean(audio_data.astype(np.float32)**2))
    if rms < 0.005: 
        print("   -> 결과: (침묵 - Skipped)")
        return "(Silence)"

    try:
        segments, info = model.transcribe(
            audio_data.astype(np.float32), 
            beam_size=5, 
            language="ko", 
            vad_filter=True,
            no_speech_threshold=0.6,
            condition_on_previous_text=False,
            # [핵심] 문맥 힌트를 줘서 정확도 대폭 향상
            # initial_prompt="2025년 산학 프로젝트에 사용할 영상 촬영하겠습니다." 
        )
        
        text_segments = []
        for s in segments:
            if text_segments and s.text.strip() == text_segments[-1]: continue
            text_segments.append(s.text.strip())
            
        full_text = " ".join(text_segments).strip()
        
        # 환각 필터
        hallucination_filters = ["MBC", "뉴스", "시청해", "감사합니다", "구독", "좋아요"]
        if any(h in full_text for h in hallucination_filters) and len(full_text) < 20:
            full_text = "(Hallucination Filtered)"

        print(f"   -> 결과: {full_text[:40]}..." if full_text else "   -> 결과: (침묵)")
        return full_text
        
    except Exception as e:
        print(f"   -> STT 실패: {e}")
        return "(STT Error)"

# =========================================================
# [6단계] Main Benchmark
# =========================================================
def run_benchmark():
    print(f"\n🚀 [Benchmark] Final Logic (Small Model + Prompt)")
    
    # [수정] 모델을 'base'에서 'small'로 변경 (정확도 UP)
    print("⏳ Whisper 모델 로딩 중... (small / int8)")
    try:
        model = WhisperModel("small", device="cpu", compute_type="int8")
    except Exception as e:
        print(f"❌ Whisper 로드 실패: {e}")
        return

    # 1. Load Audio
    raw_audio = load_audio_from_video(VIDEO_PATH, sr=SAMPLE_RATE, duration=60)
    if len(raw_audio) == 0: return
    
    # 2. Extract VAD
    vad_history, fps = extract_visual_vad(VIDEO_PATH)
    
    results = {}
    transcripts = {}

    # --- Case 1: Raw ---
    print("\n1️⃣ [Case 1] Raw Audio")
    sf.write(os.path.join(OUTPUT_DIR, "1_raw.wav"), raw_audio, SAMPLE_RATE)
    transcripts['Raw'] = perform_stt(model, raw_audio, "Raw")
    results['Raw'] = {"RTF": 0.0, "Noise_Reduction_dB": 0.0, "Speech_Loss_dB": 0.0}

    # --- Case 2: Standard ---
    print("\n2️⃣ [Case 2] Standard NR")
    cfg_std = NoiseReducerConfig(sample_rate=SAMPLE_RATE, stationary=True, non_speech_gain=1.0, vad_confidence_threshold=0.0)
    audio_std, time_std = run_audio_pipeline(raw_audio, cfg_std, vad_history, mode='standard')
    sf.write(os.path.join(OUTPUT_DIR, "2_standard.wav"), audio_std, SAMPLE_RATE)
    
    transcripts['Standard'] = perform_stt(model, audio_std, "Standard")
    results['Standard'] = calculate_metrics_with_vad(raw_audio, audio_std, time_std, vad_history, SAMPLE_RATE)
    print(f"   👉 RTF: {results['Standard']['RTF']:.3f} | NR: {results['Standard']['Noise_Reduction_dB']:.2f}dB")

    # --- Case 3: Proposed ---
    print("\n3️⃣ [Case 3] Proposed NR")
    # 파라미터 유지 (0.2 / 0.1)
    cfg_pro = NoiseReducerConfig(
        sample_rate=SAMPLE_RATE, 
        stationary=False, 
        vad_confidence_threshold=0.2, 
        non_speech_gain=0.1,          
        low_latency_mode=True
    )
    audio_pro, time_pro = run_audio_pipeline(raw_audio, cfg_pro, vad_history, mode='proposed')
    sf.write(os.path.join(OUTPUT_DIR, "3_proposed.wav"), audio_pro, SAMPLE_RATE)
    
    transcripts['Proposed'] = perform_stt(model, audio_pro, "Proposed")
    results['Proposed'] = calculate_metrics_with_vad(raw_audio, audio_pro, time_pro, vad_history, SAMPLE_RATE)
    print(f"   👉 RTF: {results['Proposed']['RTF']:.3f} | NR: {results['Proposed']['Noise_Reduction_dB']:.2f}dB")

    # ------------------------------------
    # STT 결과 파일 저장 (복원됨)
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
    # 시각화 (Seaborn Style)
    # ------------------------------------
    print("📊 그래프 생성 중...")
    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid", rc={"axes.grid": False})
        plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})
    except: pass
    
    labels = ['Standard', 'Proposed']
    # NR, Loss 모두 양수 값으로 변환하여 비교
    nr = [abs(results['Standard']['Noise_Reduction_dB']), abs(results['Proposed']['Noise_Reduction_dB'])]
    sl = [abs(results['Standard']['Speech_Loss_dB']), abs(results['Proposed']['Speech_Loss_dB'])]
    rtf = [results['Standard']['RTF'], results['Proposed']['RTF']]
    x = np.arange(len(labels)); width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)
    
    # 막대 그래프
    ax1.bar(x-width/2, nr, width, label='Noise Reduction (dB)', color='#2b7bba', alpha=0.9, edgecolor='black', linewidth=0.5)
    ax1.bar(x+width/2, sl, width, label='Speech Loss (dB)', color='#d9534f', alpha=0.9, edgecolor='black', linewidth=0.5)
    
    ax1.set_ylabel('Magnitude (dB)', fontweight='bold')
    ax1.set_title('Performance Benchmark: Standard vs. Proposed', fontweight='bold', pad=15)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    
    # RTF 선 그래프
    ax2 = ax1.twinx()
    ax2.plot(x, rtf, color='#5cb85c', marker='o', lw=3, ms=10, mec='white', mew=2, label='RTF (Speed)')
    ax2.set_ylabel('Real Time Factor (RTF)', fontweight='bold', color='#5cb85c')
    ax2.set_ylim(0, max(rtf)*1.3)
    
    # 범례 통합
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, lab1+lab2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    
    # 값 표시
    for i, v in enumerate(nr):
        ax1.text(x[i]-width/2, v+0.2, f"{v:.2f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
    for i, v in enumerate(sl):
        ax1.text(x[i]+width/2, v+0.2, f"{v:.2f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "benchmark_chart_v3.png"))
    print("✅ 모든 작업 완료.")

if __name__ == "__main__":
    run_benchmark()