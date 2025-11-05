import whisper
import os
import glob 

# --- (수정됨) 프로젝트 루트 경로를 동적으로 찾기 ---
# __file__은 현재 실행 중인 스크립트의 경로를 나타냅니다.
# os.path.abspath(__file__) -> 스크립트의 절대 경로
# os.path.dirname(...) -> 해당 파일이 속한 폴더(디렉토리) 경로
# 즉, PROJECT_ROOT는 run_stt_analysis.py가 있는 폴더가 됩니다.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def run_whisper(model, file_path):
    """
    지정된 오디오 파일에 대해 Whisper STT를 실행하고 텍스트를 반환합니다.
    """
    abs_path = os.path.abspath(file_path)
    
    # 1. 파일 존재 여부 확인
    if not os.path.exists(abs_path):
        print(f"[오류] 파일을 찾을 수 없습니다: {abs_path}")
        return "[파일 없음]"
    
    # 2. 파일 크기 확인 (0바이트 파일 방지)
    try:
        file_size = os.path.getsize(abs_path)
        if file_size == 0:
            print(f"[경고] 파일 크기가 0입니다 (빈 파일): {abs_path}")
            return "[빈 파일]"
    except Exception as e:
        print(f"[오류] 파일 크기 확인 중 오류: {e}")
        return "[파일 접근 오류]"

    # 3. Whisper 처리
    try:
        result = model.transcribe(abs_path, language="ko")
        return result["text"]
    except Exception as e:
        print(f"[오류] {abs_path} 처리 중 오류 발생: {e}")
        return f"[처리 오류: {e}]"

# --- 메인 실행 ---
if __name__ == "__main__":
    
    print("========= 성능 분석 STT 테스트 시작 =========\n")
    print("Whisper 'base' 모델을 로드합니다...")
    
    model = whisper.load_model("base")
    print("모델 로드 완료.\n")

    # --- 1. 파일 경로 설정 (상대 경로로 수정) ---

    # PROJECT_ROOT를 기준으로 하위 폴더 경로를 조합합니다.
    ORIGINAL_BASE_DIR = os.path.join(
        PROJECT_ROOT, 'audio-module', 'src', 'recognizer', 'audio', 'data', 'interim'
    )
    ORIGINAL_FILE = os.path.join(ORIGINAL_BASE_DIR, 'input_audio.wav')
    
    PROCESSED_DIR = os.path.join(
        PROJECT_ROOT, 'audio-module', 'src', 'recognizer', 'audio', 'data', 'processed'
    )

    # --- 2. '원본' 오디오 STT 실행 ---
    print("--- 1. 원본 오디오 (처리 전) STT ---")
    original_text = run_whisper(model, ORIGINAL_FILE)
    print(f"결과: {original_text}\n")

    # --- 3. '처리된' 오디오 STT 모두 실행 ---
    print("--- 2. 처리된 오디오 (처리 후) STT ---")
    
    cleaned_files_list = glob.glob(os.path.join(PROCESSED_DIR, 'cleaned_*.wav'))
    
    if not cleaned_files_list:
        print(f"[경고] {PROCESSED_DIR}에서 'cleaned_'로 시작하는 wav 파일을 찾을 수 없습니다.")
        print(" -> 경로가 올바른지, 해당 폴더에 'cleaned_'로 시작하는 파일이 있는지 확인해 주세요.")
    
    for file_path in cleaned_files_list:
        file_name = os.path.basename(file_path)
        print(f"파일: {file_name}")
        
        cleaned_text = run_whisper(model, file_path)
        print(f"결과: {cleaned_text}\n")

    print("========= STT 테스트 종료 =========")