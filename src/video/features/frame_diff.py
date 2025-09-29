import cv2
import numpy as np


def frame_difference(prev_gray, gray):
    if prev_gray is None or gray is None:
        return 0.0
    diff = cv2.absdiff(prev_gray, gray)
    
    return float(np.mean(diff) / 255.0)
