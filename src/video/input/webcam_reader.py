import cv2
import time
from config import cfg


class WebcamReader:
    def __init__(self, cam_id=0):
        self.cap = cv2.VideoCapture(cam_id)
        if not self.cap.isOpened():
            raise ValueError("웹캠을 열 수 없습니다.")
        self.start_time = time.time()  # 실시간 기준 시작 시각


    def read(self):
        ret, frame = self.cap.read()
        return self.cap.read()


    def get_timestamp(self):
        """실시간 처리: 시작 시점 기준 경과 시간 (초 단위)"""
        return time.time() - self.start_time


    def release(self):
        self.cap.release()
        