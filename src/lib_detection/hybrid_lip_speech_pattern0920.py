# hybrid : 옵티컬 플로우 + 거리 기반 특징 결합
# lip_speech : 입술 움직임으로 발화 감지
# pattern : 반복적인 입술 열림/닫힙 패턴으로 발화 여부 판단

# hybrid_lip_speech_pattern_baseline.py
import cv2
import mediapipe as mp
import numpy as np
from collections import deque

mp_face_mesh = mp.solutions.face_mesh

# 입술 랜드마크 인덱스 (MediaPipe FaceMesh 기준)
UPPER_LIP_IDX = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LOWER_LIP_IDX = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

# ====================== 조정 가능한 파라미터 ======================
# NOTE: dist는 normalized (눈 간 거리로 나누어진 값) 입니다.
LIP_THRESH = 0.15            # (비율) baseline 대비 얼마나 커지면 즉시 발화로 보는지 (예: 0.2 = 20% 증가)
MOVEMENT_THRESH = 0.02       # (정규화된 비율) 입술 이동량 임계치 (눈간거리로 정규화한 값)
PATTERN_BUFFER_LEN = 6       # 패턴 분석용 버퍼 길이 (최근 N프레임)
PATTERN_DIFF_THRESH = 0.02   # 버퍼 내에서의 per-frame 변화가 이 값보다 크면 "유의미한 변화"로 간주
PATTERN_SCORE_THRESH = 0.4   # 버퍼 내에서 유의미 변화 비율이 이 값 이상이면 패턴 감지

BASELINE_FRAMES = 60         # 시작할 때 baseline(중립) 측정용 프레임 수 (예: 60 -> 약 2초 @30fps)
MIN_SPEAK_FRAMES = 3         # 발화 시작으로 간주하려면 연속/누적 detection이 이만큼 필요
SPEAK_HOLD_LIMIT = 15        # speak_counter 최대값 (유지/히스테리시스)
# =================================================================

# 입술 거리 계산 (정규화 적용). 리턴: (normalized_dist, lips_points_array, ref_dist)
def normalized_lip_distance(landmarks, w, h):
    top_lip = np.array([[landmarks[i].x*w, landmarks[i].y*h] for i in UPPER_LIP_IDX])
    bottom_lip = np.array([[landmarks[i].x*w, landmarks[i].y*h] for i in LOWER_LIP_IDX])
    top_center = np.mean(top_lip, axis=0)
    bottom_center = np.mean(bottom_lip, axis=0)
    dist = np.linalg.norm(top_center - bottom_center)

    # === 정규화: 양쪽 눈 사이 거리 사용 (landmark 33 / 263) ===
    left_eye = np.array([landmarks[33].x*w, landmarks[33].y*h])
    right_eye = np.array([landmarks[263].x*w, landmarks[263].y*h])
    ref_dist = np.linalg.norm(left_eye - right_eye)
    if ref_dist > 1e-6:
        norm_dist = dist / ref_dist
    else:
        norm_dist = dist  # 극단적 경우 방어 코드

    lips_points = np.vstack([top_lip, bottom_lip])
    return norm_dist, lips_points, ref_dist

# ========== 초기화 ==========
cap = cv2.VideoCapture(0)
prev_lip = None
lip_buffer = deque(maxlen=PATTERN_BUFFER_LEN)
baseline_buffer = deque(maxlen=BASELINE_FRAMES)
baseline = None
speak_counter = 0

with mp_face_mesh.FaceMesh(max_num_faces=1) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        # 초기화
        is_speaking = False
        dist = 0.0
        movement = 0.0
        ref_dist = 1.0
        ratio = 1.0
        pattern_score = 0.0   
        pattern_detected = False

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            h, w, _ = frame.shape

            # normalized distance, lips points, reference distance
            dist, lips_points, ref_dist = normalized_lip_distance(landmarks, w, h)

            # calibration (baseline) 수집: 시작 시 baseline_buffer에 모음
            if len(baseline_buffer) < BASELINE_FRAMES:
                baseline_buffer.append(dist)
                baseline = np.mean(baseline_buffer)
                # show calibration progress
                cv2.putText(frame, f"Calibrating... ({len(baseline_buffer)}/{BASELINE_FRAMES})", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                cv2.putText(frame, f"Baseline: {baseline:.3f}", (10,60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,0), 2)
                # show lips box even during calibration
                if lips_points.size > 0:
                    lips_pts_int = lips_points.astype(np.int32)
                    x_min, y_min = np.min(lips_pts_int, axis=0)
                    x_max, y_max = np.max(lips_pts_int, axis=0)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (180,180,0), 2)
                cv2.imshow("Lip Movement Hybrid Pattern", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue  # calibration 하는 동안 detection은 하지 않음

            # after baseline collected
            baseline = np.mean(baseline_buffer)  # keep using mean baseline (robust option: median)

            # append to pattern buffer (use normalized dist)
            lip_buffer.append(dist)

            # draw bounding box (int32)
            lips_pts_int = lips_points.astype(np.int32)
            if lips_pts_int.size > 0:
                x_min, y_min = np.min(lips_pts_int, axis=0)
                x_max, y_max = np.max(lips_pts_int, axis=0)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # optical-flow like movement: use lip center movement, but normalize by ref_dist
            lip_center = np.mean(lips_points, axis=0)
            if prev_lip is not None:
                prev_pts = np.array(prev_lip, dtype=np.float32).reshape(-1,1,2)
                curr_pts = np.array(lips_points, dtype=np.float32).reshape(-1,1,2)
                flow = np.linalg.norm(curr_pts - prev_pts, axis=2)
                movement_px = np.mean(flow)  # pixel unit
                # normalize movement by ref_dist so threshold is scale invariant
                if ref_dist > 1e-6:
                    movement = movement_px / ref_dist
                else:
                    movement = movement_px
            else:
                movement = 0.0
            prev_lip = lips_points.copy()

            # relative increase ratio compared to baseline
            ratio = dist / (baseline + 1e-9)

            # ---------- pattern detection ----------
            pattern_detected = False
            if len(lip_buffer) == lip_buffer.maxlen:
                diffs = np.abs(np.diff(np.array(lip_buffer)))
                # pattern score = 비의미한 변화가 아닌 "유의미 변화" 발생 비율
                if len(diffs) > 0:
                    pattern_count = np.sum(diffs > PATTERN_DIFF_THRESH)
                    pattern_score = pattern_count / len(diffs)
                else:
                    pattern_score = 0.0

                # pattern_detected if either (a) many frames show changes OR (b) a single large change
                if (pattern_score >= PATTERN_SCORE_THRESH) or (np.max(diffs) > PATTERN_DIFF_THRESH * 2):
                    pattern_detected = True

            # ---------- combine rules ----------
            # instant_detection: significant relative lip opening OR significant movement
            instant_detection = (ratio > 1.0 + LIP_THRESH) or (movement > MOVEMENT_THRESH)
            # relaxed_detection: smaller relative increase OR movement + pattern together
            relaxed_detection = ((ratio > 1.0 + (LIP_THRESH / 2.0)) and pattern_detected) or \
                                ((movement > (MOVEMENT_THRESH / 2.0)) and pattern_detected)

            detection = instant_detection or relaxed_detection

            # speak_counter (hysteresis) to avoid flicker
            if detection:
                speak_counter = min(speak_counter + 1, SPEAK_HOLD_LIMIT)
            else:
                speak_counter = max(speak_counter - 1, 0)

            is_speaking = speak_counter >= MIN_SPEAK_FRAMES

            # 화면 표시 (debug)
            cv2.putText(frame, f"Baseline: {baseline:.3f}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 2)
            cv2.putText(frame, f"Dist(norm): {dist:.3f}  Ratio: {ratio:.2f}", (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(frame, f"Move(norm): {movement:.3f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Pattern: {pattern_detected}  Score: {pattern_score:.2f}", (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)
            cv2.putText(frame, f"SpeakCnt: {speak_counter}  Speaking: {is_speaking}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if is_speaking else (0, 255, 0), 2)

        else:
            # no face detected: optionally decay speak counter
            speak_counter = max(speak_counter - 1, 0)

        cv2.imshow("Lip Movement Hybrid Pattern", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
