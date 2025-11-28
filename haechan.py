# 예시 코드입니다~~ 약간 이런 방향으로 정의를 해보았습니다!!
# main_pipeline.py
# 역할: 전체 파이프라인을 총괄하며, 각 모듈을 순서대로 호출하고 데이터를 연결
import numpy as np
from moviepy.editor import VideoFileClip
import soundfile as sf # 오디오 파일을 저장하기 위한 라이브러리 (pip install soundfile)

# 현지님과 원후님이 만든 모듈을 import 합니다.
import video_module
import audio_module

def run_voice_team_pipeline(input_video_path: str, output_audio_path: str):
    """
    우리 Voice팀의 전체 파이프라인을 실행합니다.
    """
    print(" Voice팀 전체 파이프라인을 시작합니다!")

    # --- 1단계: 재료 준비  ---
    # 동영상에서 오디오 전체를 추출하여 데이터로 변환합니다.
    print(" 1단계: 동영상에서 오디오를 추출합니다...")
    video = VideoFileClip(input_video_path)
    full_audio = video.audio
    audio_data = full_audio.to_soundarray()
    sample_rate = full_audio.fps

    # --- 2단계: 영상 분석 모듈 호출 (현지님 모듈 사용) ---
    print("\n 2단계: 현지님의 영상 분석 모듈을 호출합니다...")
    speech_timestamps = video_module.detect_speech_segments(input_video_path)

    # --- 3단계: 음성 처리 모듈 호출 (원후님 모듈 사용) ---
    print("\n 3단계: 원후님의 음성 처리 모듈을 조건부로 호출합니다...")
    final_audio_data = audio_data.copy() # 최종 결과물을 담을 데이터

    for start_time, end_time in speech_timestamps:
        print(f"   - 발화 구간 [{start_time}초 ~ {end_time}초]에 잡음 제거를 적용합니다.")

        # 시간 정보를 인덱스로 변환
        start_index = int(start_time * sample_rate)
        end_index = int(end_time * sample_rate)

        # 해당 구간의 오디오 데이터만 잘라내기
        speech_chunk = final_audio_data[start_index:end_index]

        # 원후님 모듈 호출!
        denoised_chunk = audio_module.denoise_audio(speech_chunk, sample_rate)

        # 처리된 오디오를 다시 원래 위치에 붙여넣기
        final_audio_data[start_index:end_index] = denoised_chunk

    # --- 4단계: 최종 결과물 저장  ---
    print("\n💾 4단계: 최종적으로 깨끗해진 오디오를 파일로 저장합니다...")
    sf.write(output_audio_path, final_audio_data, sample_rate)
    print(f"🎉 성공! '{output_audio_path}' 파일이 생성되었습니다.")


# --- 이 파일을 직접 실행했을 때만 아래 코드가 동작합니다 ---
if __name__ == "__main__":
    # 테스트용 동영상 파일 (실제 파일 경로로 변경 필요)
    test_video = "my_test_video.mp4"
    # 결과물이 저장될 파일 이름
    output_wav = "final_clean_audio.wav"

    run_voice_team_pipeline(test_video, output_wav)