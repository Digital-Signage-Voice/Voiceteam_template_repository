import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

class LipExtractor:
    def __init__(self, static_mode=False, max_faces=1, refine_landmarks=True, min_detection_confidence=0.5):
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=static_mode,
            max_num_faces=max_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence
        )

        # 입술 랜드마크 index (Mediapipe Face Mesh 기준)
        self.upper_lip_idx = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409]
        self.lower_lip_idx = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

    def extract(self, frame):
        """
        frame: BGR 이미지
        return: dict with 'lip_points' (np.ndarray shape (N,2))
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]
        h, w, _ = frame.shape
        landmarks = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks.landmark])

        # 입술 좌표
        lip_points = np.vstack([landmarks[self.upper_lip_idx], landmarks[self.lower_lip_idx]])

        # 얼굴 중심점 (눈과 코)
        left_eye = np.mean(landmarks[[33, 133]], axis=0)   # 좌우 눈 중심
        right_eye = np.mean(landmarks[[362, 263]], axis=0)
        nose_tip = landmarks[1]

        face_ref = {'left_eye': left_eye, 'right_eye': right_eye, 'nose_tip': nose_tip}

        # 보정된 입술 비율 계산
        lip_ratio = self.lip_aspect_ratio(lip_points, face_ref)

        return {'lip_points': lip_points, 'lip_ratio': lip_ratio}

    @staticmethod
    def lip_aspect_ratio(lip_points, face_landmarks):
        # 좌우, 상하 기준점
        left = lip_points[np.argmin(lip_points[:, 0])]
        right = lip_points[np.argmax(lip_points[:, 0])]
        top = lip_points[np.argmin(lip_points[:, 1])]
        bottom = lip_points[np.argmax(lip_points[:, 1])]

        # 얼굴 yaw 계산
        eye_center = (face_landmarks['left_eye'] + face_landmarks['right_eye']) / 2
        dx = face_landmarks['right_eye'][0] - face_landmarks['left_eye'][0]
        dy = face_landmarks['right_eye'][1] - face_landmarks['left_eye'][1]
        yaw_angle = np.arctan2(dy, dx)
        cos_yaw = np.cos(yaw_angle)
        sin_yaw = np.sin(yaw_angle)

        # 좌표 회전
        def rotate_point(p):
            px, py = p - eye_center
            rx = px * cos_yaw + py * sin_yaw
            ry = -px * sin_yaw + py * cos_yaw
            return np.array([rx, ry])

        left = rotate_point(left)
        right = rotate_point(right)
        top = rotate_point(top)
        bottom = rotate_point(bottom)

        horiz = np.linalg.norm(right - left) + 1e-6
        vert = np.linalg.norm(bottom - top) + 1e-6
        return float(vert / horiz)
