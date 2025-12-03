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
        
        # [NEW] 스무딩을 위한 이전 신뢰도 저장 변수
        self.prev_confidence = 0.0

    def check_head_turn(self, det, threshold=0.20):
        """
        고개를 돌렸는지 판단
        """
        if det is None:
            return False
        
        yaw = det.get('head_yaw', 0.0)
        # Yaw 절대값이 임계값을 넘으면 회전으로 간주
        if abs(yaw) > threshold:
            print(f"[Head Turn] 감지됨 (Yaw: {yaw:.2f})") # 디버깅용
            return True
        return False
   
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
                print(f"{RED}[Warning] LipExtractor 오류 (ROI): {e}{RESET}")

        # 3-1️⃣ Fallback: ROI에서 실패했거나 사람이 감지되지 않은 경우 전체 프레임에서 시도
        if det is None:
            try:
                # 전체 프레임에서 시도 (속도는 느릴 수 있음)
                det = self.extractor.extract(frame)
                if det:
                    lip_pts_np = det['lip_points']
                    # 전체 프레임 기준이므로 좌표 변환 필요 없음
                    ratio = det.get('lip_ratio', 0.0)
            except Exception as e:
                pass # Fallback도 실패하면 무시

        # 고개 돌림 확인
        is_head_turning = self.check_head_turn(det, threshold=0.35)

        # 4️⃣ 프레임 차이 계산
        diff_val = frame_difference(self.prev_gray, gray, lip_pts_np)

        # 5️⃣ 분류 (Speaking 여부 판단)
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

        if main_target:
            detection_conf = main_target['conf']  # YOLOv5 출력값
        else:
            detection_conf = 0.0
        feature_quality = self.calc_feature_quality(det)
        lip_ratio = self.calc_lip_ratio(det)
        
        # lip_ratio 보정 logic
        if lip_ratio > 0.8:
            ratio_for_conf = 0.65
        elif lip_ratio > 0.65:
            ratio_for_conf = 0.9 * lip_ratio
        else:
            ratio_for_conf = lip_ratio

        # 8️⃣ Confidence 계산
        # [A] Score Calculation
        # [수정] 움직임 민감도 하향 (8.0 -> 5.0)
        score_diff = min(diff_val * 5.0, 1.0)
        
        # 입이 조금이라도(0.2) 열리면 점수 부여
        score_ratio = 0.0
        if ratio_for_conf > 0.2:
            score_ratio = min((ratio_for_conf - 0.2) * 5.0, 1.0)

        # [B] Raw Confidence Calculation
        # [수정] 가중치 역전: 움직임(0.4) < 입모양(0.6)
        # 이제 입을 다물고 움직이면(Diff=1.0, Ratio=0.0) -> 0.4점밖에 못 받음 (비발화)
        raw_conf = (0.4 * score_diff) + (0.6 * score_ratio)

        # [C] Boolean Boost
        if speaking:
            # 말하고 있으면 기본 점수 보장
            raw_conf = max(raw_conf, 0.65) + 0.1
        
        # [D] Smoothing
        # 점수가 갑자기 떨어지는 것을 방지 (이전 값 70% 반영)
        combined_confidence = (0.3 * raw_conf) + (0.7 * self.prev_confidence)

        # [E] Safety Gating
        if not person_detected or feature_quality < 0.2:
            combined_confidence = 0.0

        # [F] 고개 돌림 감지 시 강제 차단
        if is_head_turning:
            combined_confidence = 0.0
            speaking = False

        # 범위 제한 (0.0 ~ 1.0)
        combined_confidence = min(max(combined_confidence, 0.0), 1.0)
        
        # 다음 프레임을 위해 저장
        self.prev_confidence = combined_confidence

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
            "person_detected": person_detected
        }

        return result

    
    def run(self):
        print(f"{YELLOW}영상 처리 프로세서 실행 중... 잠시만 기다려주세요.{RESET}")
    
        frame_id = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("비디오 소스 종료 또는 읽기 실패")
                break

            try:
                result = self.process_frame(frame_id, frame)
                
                # 결과 출력
                # print(result)

                if self.visualize:
                    frame_vis = Overlay.draw(frame.copy(), result)
                    
                    # 리사이즈 및 출력
                    h, w = frame_vis.shape[:2]
                    scale = min(cfg.window_width / w, cfg.window_height / h, 1.0)
                    new_w, new_h = int(w * scale), int(h * scale)
                    frame_vis_resized = cv2.resize(frame_vis, (new_w, new_h))
                    
                    cv2.imshow(cfg.window_name, frame_vis_resized)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
            
            except Exception as e:
                # 에러가 발생해도 죽지 않고 로그만 남기고 계속 실행
                print(f"{RED}[Error] Frame skipping due to error: {e}{RESET}")
                continue

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