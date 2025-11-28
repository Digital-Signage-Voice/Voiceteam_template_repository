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
        if w == 0 or h == 0:
            return 0.0
        prev_crop = prev_gray[y:y+h, x:x+w]
        gray_crop = gray[y:y+h, x:x+w]
    else:
        prev_crop = prev_gray
        gray_crop = gray

    if prev_crop.shape != gray_crop.shape:
        # 크기가 다르면 (예: ROI가 이미지 경계를 벗어남) 리사이즈 또는 0 반환
        return 0.0

    diff = cv2.absdiff(prev_crop, gray_crop)
    if diff is None or diff.size == 0:
        return 0.0
        
    mean_val = np.mean(diff)
    if mean_val is None:
        return 0.0
        
    return float(mean_val / 255.0)
