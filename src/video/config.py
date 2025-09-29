from dataclasses import dataclass


@dataclass
class Config:
    lip_ratio_threshold: float = 0.40
    open_frames_to_speaking: int = 3
    flow_magnitude_threshold: float = 0.8
    frame_width: int = 640
    frame_height: int = 480
    model_path: str = "ml_lip_svm.joblib"
    max_queue_len: int = 5


cfg = Config()