# hybrid : 옵티컬 플로우 + 거리 기반 특징 결합
# lip_speech : 입술 움직임으로 발화 감지
# pattern : 반복적인 입술 열림/닫힙 패턴으로 발화 여부 판단

import cv2
import mediapipe as mp
import numpy as np
from collections import deque

mp_face_mesh = mp.solutions.face_mesh

# 입술 랜드마크 인덱스
UPPER_LIP_IDX = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LOWER_LIP_IDX = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

# ====================== 조정 가능한 파라미터 ======================
LIP_THRESH = 15.0           # 입술 거리 임계치 - 입이 얼마나 벌어졌는지
MOVEMENT_THRESH = 1.0       # 옵티컬 플로우 이동량 임계치 - 입술 좌표 이동량
PATTERN_BUFFER_LEN = 5      # 반복 패턴 분석용 버퍼 길이
PATTERN_DIFF_THRESH = 0.5   # 반복 패턴 차이 임계치 - 버퍼 내 거리 변화가 일정 수준 반복되는지
# =================================================================

# 입술 거리 계산
def lip_distance(landmarks, w, h):
    top_lip = np.array([[landmarks[i].x*w, landmarks[i].y*h] for i in UPPER_LIP_IDX])
    bottom_lip = np.array([[landmarks[i].x*w, landmarks[i].y*h] for i in LOWER_LIP_IDX])
    top_center = np.mean(top_lip, axis=0)
    bottom_center = np.mean(bottom_lip, axis=0)
    return np.linalg.norm(top_center - bottom_center), np.vstack([top_lip, bottom_lip])

# 비디오 캡처
cap = cv2.VideoCapture(0)
prev_lip = None
lip_buffer = deque(maxlen=PATTERN_BUFFER_LEN)

with mp_face_mesh.FaceMesh(max_num_faces=1) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        is_speaking = False
        dist = 0
        movement = 0

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            h, w, _ = frame.shape

            # 입술 거리와 좌표
            dist, lips_points = lip_distance(landmarks, w, h)
            lip_buffer.append(dist)

            # 바운딩 박스
            x_min, y_min = np.min(lips_points, axis=0).astype(int)
            x_max, y_max = np.max(lips_points, axis=0).astype(int)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # 옵티컬 플로우 기반 이동량
            if prev_lip is not None:
                prev_pts = np.array(prev_lip, dtype=np.float32).reshape(-1,1,2)
                curr_pts = np.array(lips_points, dtype=np.float32).reshape(-1,1,2)
                flow = np.linalg.norm(curr_pts - prev_pts, axis=2)
                movement = np.mean(flow)
            prev_lip = lips_points.copy()

            # 반복적인 패턴 기반 발화 판단
            if len(lip_buffer) == lip_buffer.maxlen:
                diffs = np.diff(lip_buffer)
                if (np.any(diffs > PATTERN_DIFF_THRESH) and np.any(diffs < -PATTERN_DIFF_THRESH)) \
                        and (dist > LIP_THRESH or movement > MOVEMENT_THRESH):
                    is_speaking = True

            # 화면 출력
            cv2.putText(frame, f'Dist: {dist:.2f}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f'Movement: {movement:.2f}', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f'Speaking: {is_speaking}', (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if is_speaking else (0, 255, 0), 2)

        cv2.imshow("Lip Movement Hybrid Pattern", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()


