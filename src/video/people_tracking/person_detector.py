import torch
import cv2

class PersonDetector:
    """
    YOLOv5 기반 사람 탐지 모듈 (lightweight)
    """
    def __init__(self, model_name='yolov5n', device=None, conf_thres=0.4):
        """
        Args:
            model_name: 사용할 YOLOv5 모델 (yolov5n, yolov5s, yolov5m ...)
            device: 'cuda' 또는 'cpu'
            conf_thres: confidence threshold
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.hub.load(
            'ultralytics/yolov5', model_name, pretrained=True
        ).to(self.device)
        self.model.conf = conf_thres
        self.model.classes = [0]  # 0번 클래스는 'person'

    def detect(self, frame):
        """
        Args:
            frame: OpenCV BGR frame
        Returns:
            detections: [{'bbox': [x1, y1, x2, y2], 'conf': float}, ...]
        """
        # BGR → RGB 변환
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(img, size=640)

        detections = []
        for *box, conf, cls in results.xyxy[0].cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'conf': float(conf)
            })
        return detections
