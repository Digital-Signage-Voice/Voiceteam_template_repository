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
        self.key_ids = [61, 191, 78, 308, 14, 13]
        # 코(1), 우측광대(234), 좌측광대(454)
        self.head_pose_ids = [1, 234, 454]

    def _calc_head_yaw(self, pts):
        """
        3개의 랜드마크를 이용해 얼굴의 좌우 회전(Yaw) 비율 계산
        반환값: 0.0(정면) ~ ±1.0(완전 측면)
        """
        try:
            nose = np.array(pts[1])
            right_edge = np.array(pts[234]) # 화면상 왼쪽
            left_edge = np.array(pts[454])  # 화면상 오른쪽
            
            face_width = np.linalg.norm(left_edge - right_edge)
            if face_width == 0: return 0.0

            face_center = (left_edge + right_edge) / 2
            yaw_diff = nose[0] - face_center[0]
            
            # 정규화된 Yaw 값
            return yaw_diff / (face_width / 2)
        except:
            return 0.0

    def extract(self, frame, offset=(0,0)):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        if results.multi_face_landmarks is None:
            return None

        face_landmarks = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]

        def to_global(l):
            return (int(l.x * w) + offset[0], int(l.y * h) + offset[1])

        pts = [to_global(l) for l in face_landmarks.landmark]
        lip_points = [pts[idx] for idx in self.key_ids if idx < len(pts)]

        lip_ratio = 0.0
        if len(lip_points) >= 6:
            top = lip_points[4]
            bottom = lip_points[5]
            left = lip_points[0]
            right = lip_points[3]
            width = np.linalg.norm(np.array(left)-np.array(right))
            height = np.linalg.norm(np.array(top)-np.array(bottom))
            lip_ratio = max(0.01, height / max(1.0, width))

        head_yaw = self._calc_head_yaw(pts)

        return {
            'landmarks': pts,
            'lip_points': np.array(lip_points, dtype=np.int32),
            'lip_ratio': lip_ratio,
            'head_yaw': head_yaw
        }