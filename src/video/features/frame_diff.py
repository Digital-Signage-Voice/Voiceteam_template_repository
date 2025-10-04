import cv2
import numpy as np

def frame_difference(prev_gray, gray, lip_points=None):
    """
    prev_gray : 이전 프레임 (grayscale)
    gray      : 현재 프레임 (grayscale)
    lip_points: np.ndarray shape(N,2), 입술 좌표. None이면 전체 프레임 사용

    반환 : 입술 영역 평균 픽셀 변화값 (0~1)
    """
    if prev_gray is None or gray is None:
        return 0.0

    if lip_points is not None and len(lip_points) > 0:
        # 입술 bounding box 계산
        x, y, w, h = cv2.boundingRect(lip_points.astype(np.int32))
        prev_crop = prev_gray[y:y+h, x:x+w]
        gray_crop = gray[y:y+h, x:x+w]
    else:
        prev_crop = prev_gray
        gray_crop = gray

    diff = cv2.absdiff(prev_crop, gray_crop)
    return float(np.mean(diff) / 255.0)
