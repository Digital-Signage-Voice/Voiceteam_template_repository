# lip_movement_hybrid_full.py
import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

UPPER_LIP_IDX = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LOWER_LIP_IDX = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

# 입술 거리
def lip_distance(landmarks, w, h):
    top_lip = np.array([[landmarks[i].x*w, landmarks[i].y*h] for i in UPPER_LIP_IDX])
    bottom_lip = np.array([[landmarks[i].x*w, landmarks[i].y*h] for i in LOWER_LIP_IDX])
    
    top_center = np.mean(top_lip, axis=0)
    bottom_center = np.mean(bottom_lip, axis=0)
    
    return np.linalg.norm(top_center - bottom_center)

cap = cv2.VideoCapture(0)
prev_gray = None
prev_lip = None
movement_thresh = 2.0
lip_thresh = 20.0  # 입술 거리 임계치

with mp_face_mesh.FaceMesh(max_num_faces=1) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        is_speaking = False
        dist = 0

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            h, w, _ = frame.shape

            # 입술 좌표
            lip_coords = np.array([[int(landmarks[i].x*w), int(landmarks[i].y*h)] for i in UPPER_LIP_IDX + LOWER_LIP_IDX])

            # 바운딩 박스
            x_min, y_min = np.min(lip_coords, axis=0)
            x_max, y_max = np.max(lip_coords, axis=0)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # 입술 거리 기반
            dist = lip_distance(landmarks, w, h)

            # 옵티컬 플로우 기반 이동량
            if prev_gray is not None and prev_lip is not None:
                prev_pts = np.array(prev_lip, dtype=np.float32).reshape(-1,1,2)
                curr_pts = np.array(lip_coords, dtype=np.float32).reshape(-1,1,2)
                flow = np.linalg.norm(curr_pts - prev_pts, axis=2)
                movement = np.mean(flow)
                if dist > lip_thresh or movement > movement_thresh:
                    is_speaking = True
            prev_gray = gray.copy()
            prev_lip = lip_coords.copy()

            # 텍스트 출력
            cv2.putText(frame, f'Dist: {dist:.2f}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f'Speaking: {is_speaking}', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if is_speaking else (0, 255, 0), 2)

        cv2.imshow("Lip Movement Hybrid Full", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
