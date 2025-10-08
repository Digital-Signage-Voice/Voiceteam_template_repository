import cv2
from config import cfg


class WebcamReader:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)


    def read(self):
        return self.cap.read()


    def release(self):
        self.cap.release()
        