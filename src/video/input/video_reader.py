import cv2
from config import cfg


class VideoReader:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise ValueError(f"비디오 파일을 열 수 없습니다: {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = 0


    def read(self):
        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
        return ret, frame


    def get_timestamp(self):
        """프레임 번호 기준 타임스탬프 계산 (초 단위)"""
        if self.cap.get(cv2.CAP_PROP_POS_FRAMES) == 0:
            return 0.0
        return self.frame_count / self.fps


    def release(self):
        self.cap.release()
        