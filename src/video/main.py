import numpy as np
import warnings
import argparse
import cv2
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


def parse_args():
    """
    명령줄 인자 parser 함수.
    실행 시 옵션을 통해 입력 소스와 분류 방식을 선택할 수 있음.

    Returns:
        argparse.Namespace:
            - source (str): 입력 소스 선택 ('webcam' 또는 'video')
            - path (str): 비디오 파일 경로 (source='video'일 때 필요)
            - use_ml (bool): True면 머신러닝(SVM) 기반 분류 사용, False면 Rule 기반 사용
            
    예시:
        # 웹캠 입력 + 규칙 기반 분류
        python main.py --source webcam

        # 비디오 파일 입력 + 규칙 기반 분류
        python main.py --source video --path sample.mp4

        # 웹캠 입력 + 머신러닝 분류기 사용
        python main.py --source webcam --use_ml        
    """

    p = argparse.ArgumentParser()
    p.add_argument('--source', choices=['webcam', 'video'], default='webcam')
    p.add_argument('--path', type=str, default=None)
    p.add_argument('--use_ml', action='store_true')
    
    return p.parse_args()


def main():
    args = parse_args()
    
    # 실행 시작 알림
    print(f"{YELLOW}프로토타입 실행 중... 잠시만 기다려주세요.{RESET}")
    
    cap = WebcamReader() if args.source == 'webcam' else VideoReader(args.path)
    extractor = LipExtractor()
    rule_cls = RuleBasedClassifier()
    ml_cls = MLClassifier() 
    
    if args.use_ml:
        print(f"{YELLOW}ML 모델 로드 중... 잠시만 기다려주세요.{RESET}")
        ml_cls = MLClassifier()
        print(f"{YELLOW}ML 모델 로드 완료!{RESET}")

    prev_gray, prev_lip_pts = None, None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        det = extractor.extract(frame)
        lip_pts_np = det['lip_points'] if det else None
        ratio = det['lip_ratio'] if det else 0.0
        
        diff_val = frame_difference(prev_gray, gray, lip_pts_np)
        
        if args.use_ml and ml_cls is not None:
            print(f"{YELLOW}ML 모델 예측 중...{RESET}", end='\r') 

            feat = [ratio or 0.0, diff_val]
            speaking = ml_cls.predict(feat)
        else:
            speaking = rule_cls.predict(ratio or 0.0, diff_val)

        Overlay.draw(frame, lip_pts_np if lip_pts_np is not None else [], ratio or 0.0, diff_val, speaking)

        # 창 설정 (고정 크기)
        cv2.namedWindow(cfg.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(cfg.window_name, cfg.window_width, cfg.window_height)

        cv2.imshow(cfg.window_name, frame)

        prev_gray = gray.copy()
        prev_lip_pts = lip_pts_np.astype(np.float32) if lip_pts_np is not None else None

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
    