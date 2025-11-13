import cv2

class Overlay:
    @staticmethod
    def draw(frame, result):
        """
        영상 프레임 위에 분석 결과 시각화
        result: VideoProcessor.process_frame() 반환 dict
        """
        if result is None:
            return frame

        h, w, _ = frame.shape
        font_scale = max(0.8, min(2.0, w / 1000))

        # ---------------------------
        # 1️⃣ ROI (입술 영역)
        # ---------------------------
        roi = result.get("roi", None)
        if roi:
            x, y, rw, rh = roi["x"], roi["y"], roi["w"], roi["h"]
            color = (0, 255, 0) if result.get("is_speaking") else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + rw, y + rh), color, 2)

        # ---------------------------
        # 2️⃣ 텍스트 정보
        # ---------------------------
        fid = result.get("frame_id", 0)
        ts = result.get("timestamp", 0.0)
        conf = result.get("confidence", 0.0)
        speaking = result.get("is_speaking", False)
        person_detected = result.get("person_detected", False)
        flags = result.get("flags", {})

        text_color = (255, 255, 255)
        line_height = 25
        y0 = 30
        
        # Confidence 표시 (색상 바)
        conf_bar_x = 10
        conf_bar_y = y0 + 2 * line_height + 5
        conf_bar_w = int(conf * 200)
        cv2.rectangle(frame, (conf_bar_x, conf_bar_y), (conf_bar_x + 200, conf_bar_y + 10), (100, 100, 100), -1)
        cv2.rectangle(frame, (conf_bar_x, conf_bar_y), (conf_bar_x + conf_bar_w, conf_bar_y + 10), (0, 255, 255), -1)
        cv2.putText(frame, f"Confidence: {conf:.2f}", (10, conf_bar_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)

        # 발화 상태
        cv2.putText(frame, f"Speaking: {'YES' if speaking else 'NO'}", (10, conf_bar_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if speaking else (0, 0, 255), 2)

        # 사람 탐지 여부
        cv2.putText(frame, f"Person detected: {person_detected}", (10, conf_bar_y + 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255) if person_detected else (100, 100, 100), 1)

        # Flags 정보
        flag_text = f"Flags: roi={flags.get('roi_detected', False)}, ts={flags.get('timestamp_valid', False)}"
        cv2.putText(frame, flag_text, (10, conf_bar_y + 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # ---------------------------
        # 3️⃣ 시각적 상태 표시
        # ---------------------------
        if speaking:
            cv2.circle(frame, (w - 40, 40), 10, (0, 255, 0), -1)
        elif person_detected:
            cv2.circle(frame, (w - 40, 40), 10, (0, 255, 255), -1)
        else:
            cv2.circle(frame, (w - 40, 40), 10, (0, 0, 255), -1)

        return frame
