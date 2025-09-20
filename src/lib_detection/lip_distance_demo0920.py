import cv2
import mediapipe as mp
import numpy as np

mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

THRESHOLD = 5.0  # 입을 열었다고 판단할 거리 기준
prev_dist = 0

cap = cv2.VideoCapture(0)

with mp_face.FaceMesh(max_num_faces=1) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # 입술 좌표
            top_lip = np.array([[landmarks[i].x * frame.shape[1], landmarks[i].y * frame.shape[0]] for i in [13,14,15]])
            bottom_lip = np.array([[landmarks[i].x * frame.shape[1], landmarks[i].y * frame.shape[0]] for i in [17,18,19]])

            # 거리 계산
            dist = np.mean(np.linalg.norm(top_lip - bottom_lip, axis=1))

            # 발화 감지 (거리 + 변화량)
            is_speaking = dist > THRESHOLD and abs(dist - prev_dist) > 0.5
            prev_dist = dist

            # 화면 표시
            cv2.putText(frame, f"Dist: {dist:.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(frame, f"Speaking: {is_speaking}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            # 입술 위치 그리기
            for (x,y) in np.vstack((top_lip, bottom_lip)):
                cv2.circle(frame, (int(x), int(y)), 2, (255,0,0), -1)

        cv2.imshow("Lip Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
