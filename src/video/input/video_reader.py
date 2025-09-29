import cv2
from config import cfg


class VideoReader:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)


    def read(self):
        return self.cap.read()


    def release(self):
        self.cap.release()
        