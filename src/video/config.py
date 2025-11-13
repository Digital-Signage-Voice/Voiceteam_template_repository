from dataclasses import dataclass


@dataclass
class Config:
    # ====== 분류 임계값 관련 ======
    lip_ratio_threshold: float = 0.40   # 입술 비율
    open_frames_to_speaking: int = 3    # 입이 연속으로 열린 프레임 수
    
    # ====== 입력 프레임 크기 ======
    frame_width: int = 640    # 카메라/비디오 프레임 가로 크기
    frame_height: int = 480    # 카메라/비디오 프레임 세로 크기
    
    # ====== 머신러닝 관련 ======
    model_path: str = "ml_lip_svm.joblib"    # 학습된 SVM 모델 경로
    max_queue_len: int = 5    # 최근 프레임 특징값 큐 최대 길이 (평활하/안정화 목적)
    
    # ====== 시각화(창 관련) ======
    window_name: str = "Prototype"    # 창 이름
    window_width: int = 1080   # 가로형  
    window_height: int = 720   # 세로형 


cfg = Config()