import numpy as np
import cv2
from .face_detector import FaceDetector


class LipExtractor:
    def __init__(self):
        self.detector = FaceDetector()
        self.key_ids = [61, 191, 78, 308, 14, 13]


    def extract(self, frame):
        results = self.detector.detect(frame)
        if results is None:
            return None
        h, w = frame.shape[:2]
        pts = [(int(l.x * w), int(l.y * h)) for l in results.landmark]
        lip_points = [pts[idx] for idx in self.key_ids if idx < len(pts)]
        return {
            'landmarks': pts,
            'lip_points': np.array(lip_points, dtype=np.int32)
        }
