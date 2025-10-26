import cv2
import numpy as np
import mediapipe as mp

class LipExtractor:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # 입술 랜드마크 인덱스 (Mediapipe 기준)
        self.key_ids = [61, 191, 78, 308, 14, 13]

    def extract(self, frame, offset=(0,0)):
        """
        frame: ROI 이미지
        offset: ROI 좌표 시작점 (x, y), global 좌표 변환용
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        if results.multi_face_landmarks is None:
            return None

        face_landmarks = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]

        # ROI 기준 -> global frame 좌표 변환
        pts = [(int(l.x * w) + offset[0], int(l.y * h) + offset[1]) for l in face_landmarks.landmark]
        lip_points = [pts[idx] for idx in self.key_ids if idx < len(pts)]

        # lip_ratio 계산 (세로 대비 가로 비율)
        lip_ratio = 0.0
        if len(lip_points) >= 6:
            top = lip_points[4]
            bottom = lip_points[5]
            left = lip_points[0]
            right = lip_points[3]
            lip_ratio = max(0.01, np.linalg.norm(np.array(top)-np.array(bottom)) /
                                  np.linalg.norm(np.array(left)-np.array(right)))

        return {
            'landmarks': pts,
            'lip_points': np.array(lip_points, dtype=np.int32),
            'lip_ratio': lip_ratio
        }
