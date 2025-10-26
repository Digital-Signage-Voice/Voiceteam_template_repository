import cv2
import numpy as np
import warnings
from config import cfg
from input.webcam_reader import WebcamReader
from input.video_reader import VideoReader
from features.lip_ratio import LipExtractor
from features.frame_diff import frame_difference
from people_tracking.person_detector import PersonDetector
from people_tracking.simple_tracker import SimpleTracker
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

        # 입력
        self.cap = WebcamReader() if source == "webcam" else VideoReader(path)
        
        # 모듈
        self.extractor = LipExtractor()
        self.detector = PersonDetector(model_name='yolov5n')
        self.tracker = SimpleTracker()
        self.rule_cls = RuleBasedClassifier()
        self.ml_cls = MLClassifier() if use_ml else None

        # 이전 프레임 정보
        self.prev_gray = None
        self.prev_lip_pts = None

   
    def process_frame(self, frame_id, frame):
        """단일 프레임 처리 후 결과 dict 반환"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 초기화
        det = None
        person_detected = False
        lip_pts_np = None
        ratio = 0.0

        # 1️⃣ 사람 탐지
        try:
            detections = self.detector.detect(frame)
            person_detected = len(detections) > 0 if detections else False
        except Exception as e:
            print(f"[Warning] PersonDetector 오류: {e}")
            detections = []

        # 2️⃣ 단일 발화자 추정 (현재 프레임에서 가장 큰 사람 선택)
        main_target = None
        if detections:
            main_target = max(detections, key=lambda d: (d['bbox'][2]-d['bbox'][0])*(d['bbox'][3]-d['bbox'][1]))

        # 3️⃣ LipExtractor 적용 (ROI 기준)
        if main_target:
            x1, y1, x2, y2 = map(int, main_target['bbox'])
            roi_frame = frame[y1:y2, x1:x2]
            try:
                det = self.extractor.extract(roi_frame)
                if det:
                    lip_pts_np = det['lip_points']
                    # ROI 좌표를 원본 프레임 좌표로 변환
                    lip_pts_np[:, 0] += x1
                    lip_pts_np[:, 1] += y1
                    ratio = det.get('lip_ratio', 0.0)
            except Exception as e:
                print(f"{RED}[Warning] LipExtractor 오류: {e}{RESET}")

        # 4️⃣ 프레임 차이 계산
        diff_val = frame_difference(self.prev_gray, gray, lip_pts_np)

        # 5️⃣ 분류
        if self.use_ml and self.ml_cls is not None:
            feat = [ratio or 0.0, diff_val]
            speaking = self.ml_cls.predict(feat)
        else:
            speaking = self.rule_cls.predict(ratio or 0.0, diff_val)

        # 6️⃣ 이전 프레임 갱신
        self.prev_gray = gray.copy()
        self.prev_lip_pts = lip_pts_np.astype(np.float32) if lip_pts_np is not None else None

        # 7️⃣ ROI dict
        if lip_pts_np is not None:
            x, y = lip_pts_np[:, 0].min(), lip_pts_np[:, 1].min()
            w, h = lip_pts_np[:, 0].max() - x, lip_pts_np[:, 1].max() - y
            roi_dict = {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
        else:
            roi_dict = None

        # ✅ 신뢰도 계산
        # detection_conf = self.calc_detection_conf(detections, main_target)
        
        if main_target:
            detection_conf = main_target['conf']  # YOLOv5 출력값
        else:
            detection_conf = 0.0
        feature_quality = self.calc_feature_quality(det)
        lip_ratio = self.calc_lip_ratio(det)
        
        combined_confidence = (
            0.5 * detection_conf +
            0.3 * feature_quality +
            0.2 * lip_ratio
        )

        combined_confidence = min(max(combined_confidence, 0.0), 1.0)

        # 9️⃣ flags
        flags = {
            "roi_detected": det is not None,
            "timestamp_valid": self.cap.get_timestamp() is not None
        }

        result = {
            "frame_id": frame_id,
            "timestamp": self.cap.get_timestamp(),
            "roi": roi_dict,
            "is_speaking": bool(speaking),
            "confidence": round(float(combined_confidence), 3),
            "flags": flags,
            "person_detected": person_detected
        }

        return result


    
    def run(self):
        print(f"{YELLOW}영상 처리 프로세서 실행 중... 잠시만 기다려주세요.{RESET}")
    
        frame_id = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            result = self.process_frame(frame_id, frame)
            print(result)

            if self.visualize:
                frame_vis = Overlay.draw(frame.copy(), result)
                
                # 원본 비율 유지하면서 리사이즈
                h, w = frame_vis.shape[:2]
                scale = min(cfg.window_width / w, cfg.window_height / h, 1.0)  # 1.0 이상 확대 금지
                new_w, new_h = int(w * scale), int(h * scale)
                frame_vis_resized = cv2.resize(frame_vis, (new_w, new_h))
                
                cv2.imshow(cfg.window_name, frame_vis_resized)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_id += 1

        self.cap.release()
        if self.visualize:
            cv2.destroyAllWindows()
            

    def calc_detection_conf(self, detections, main_target):
        """
        사람 검출 신뢰도 계산
        - detection 결과가 있으면 0.7~1.0 사이
        - 없으면 0.0
        """
        if not detections:
            return 0.0
        if main_target is None:
            return 0.0
        # bbox 면적 비율을 confidence로 단순 변환 (예: 화면 대비)
        bbox = main_target['bbox']
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        frame_area = cfg.frame_width * cfg.frame_height
        conf = min(1.0, max(0.3, area / frame_area * 5))  # 과도한 값 방지
        return round(conf, 3)

    def calc_feature_quality(self, det):
        """
        LipExtractor의 랜드마크 품질 측정
        - 검출된 점 개수, 값 유효성 등으로 품질 측정
        """
        if det is None:
            return 0.0
        lip_pts = det.get("lip_points", None)
        if lip_pts is None or len(lip_pts) == 0:
            return 0.0
        # 점 개수 기반 간단한 평가 (정상적이면 0.8 이상)
        quality = min(1.0, len(lip_pts) / 20.0)
        return round(quality, 3)

    def calc_lip_ratio(self, det):
        """
        이미 LipExtractor에서 계산된 lip_ratio를 그대로 사용
        """
        if det is None:
            return 0.0
        return round(det.get("lip_ratio", 0.0), 3)

