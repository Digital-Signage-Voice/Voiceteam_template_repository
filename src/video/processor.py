import cv2
import numpy as np
import warnings
from config import cfg
from input.webcam_reader import WebcamReader
from input.video_reader import VideoReader
from features.lip_ratio import LipExtractor
from features.frame_diff import frame_difference
from classification.rule_based import RuleBasedClassifier
from classification.ml_model import MLClassifier
from visualization.overlay import Overlay

warnings.filterwarnings("ignore", category=UserWarning)

# 글씨 색
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
RESET = "\033[0m" 

class VideoProcessor:
    def __init__(self, source="webcam", path=None, use_ml=False, visualize=True):
        """
        영상 처리기 초기화
        """
        self.source = source
        self.path = path
        self.use_ml = use_ml
        self.visualize = visualize

        self.cap = WebcamReader() if source == "webcam" else VideoReader(path)
        self.extractor = LipExtractor()
        self.rule_cls = RuleBasedClassifier()
        self.ml_cls = MLClassifier() if use_ml else None

        self.prev_gray = None
        self.prev_lip_pts = None

    def process_frame(self, frame_id, frame):
        """단일 프레임 처리 후 결과 dict 반환"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        det = self.extractor.extract(frame)
        lip_pts_np = det['lip_points'] if det else None
        ratio = det['lip_ratio'] if det else 0.0

        diff_val = frame_difference(self.prev_gray, gray, lip_pts_np)

        # 분류
        if self.use_ml and self.ml_cls is not None:
            feat = [ratio or 0.0, diff_val]
            speaking = self.ml_cls.predict(feat)
        else:
            speaking = self.rule_cls.predict(ratio or 0.0, diff_val)


        # 이전 프레임 갱신
        self.prev_gray = gray.copy()
        self.prev_lip_pts = lip_pts_np.astype(np.float32) if lip_pts_np is not None else None

        # ROI dict
        # lip_points 좌표를 이용해서 좌표를 계산
        if det and det['lip_points'] is not None:
            points = det['lip_points']
            x, y = points[:, 0].min(), points[:, 1].min()
            w, h = points[:, 0].max() - x, points[:, 1].max() - y
            roi_dict = {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
        else:
            roi_dict = None

        # 1차 발화 구간 임시 표시 (segment_first_pass=True)
        speech_segment = []
        if speaking:
            ts = self.cap.get_timestamp() or 0.0
            speech_segment.append({"start": ts, "end": None, "verified": False})

        # flags
        flags = {
            "roi_detected": det is not None,
            "timestamp_valid": self.cap.get_timestamp() is not None,
            "segment_first_pass": speaking,
            "segment_verified": False  # 2차 검증은 동기화 모듈에서 처리
        }

        result = {
            "frame_id": frame_id,
            "timestamp": self.cap.get_timestamp(),
            "roi": roi_dict,
            "is_speaking": bool(speaking),
            "speech_segments": speech_segment,
            "flags": flags
        }

        return result

    def run(self):
        # 실행 시작 알림
        print(f"{YELLOW}영상 처리 프로세서 실행 중... 잠시만 기다려주세요.{RESET}")
    
        """영상 스트림 전체 처리"""
        frame_id = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            result = self.process_frame(frame_id, frame)

            # 개발용 dict 출력
            print(result)

            # 시각화 모드일 경우
            if self.visualize:
                # frame 위에 Overlay 그리기
                frame_vis = Overlay.draw(frame.copy(), result)
                cv2.imshow(cfg.window_name, frame_vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_id += 1

        self.cap.release()
        if self.visualize:
            cv2.destroyAllWindows()
