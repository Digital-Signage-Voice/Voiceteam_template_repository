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
        # 화면 크기에 비례한 폰트 크기 조절
        font_scale = max(0.6, min(2.0, w / 1000))
        thickness = 1 if w < 640 else 2

        # ---------------------------
        # 1️⃣ ROI (입술 영역)
        # ---------------------------
        roi = result.get("roi", None)
        if roi:
            x, y, rw, rh = roi["x"], roi["y"], roi["w"], roi["h"]
            # 말하고 있으면 초록색, 아니면 빨간색
            color = (0, 255, 0) if result.get("is_speaking") else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + rw, y + rh), color, 2)

        # ---------------------------
        # 2️⃣ 데이터 추출
        # ---------------------------
        fid = result.get("frame_id", 0)
        ts = result.get("timestamp", 0.0)
        if ts is None: ts = 0.0
        
        conf = result.get("confidence", 0.0)
        speaking = result.get("is_speaking", False)
        person_detected = result.get("person_detected", False)
        flags = result.get("flags", {})

        # 타임스탬프 포맷팅 (MM:SS.ms)
        minutes = int(ts // 60)
        seconds = int(ts % 60)
        millis = int((ts * 1000) % 1000)
        time_str = f"{minutes:02d}:{seconds:02d}.{millis:03d}"

        # ---------------------------
        # 3️⃣ 정보 텍스트 그리기
        # ---------------------------
        text_color = (255, 255, 255)
        bg_color = (0, 0, 0)
        line_height = int(30 * font_scale)
        start_x = 10
        start_y = 30

        # 정보 리스트 (라벨, 값, 색상)
        info_lines = [
            (f"Frame: {fid}", (200, 200, 200)),
            (f"Time:  {time_str}", (200, 200, 200)),
            (f"Person: {person_detected}", (0, 200, 255) if person_detected else (100, 100, 100)),
            (f"Speaking: {'YES' if speaking else 'NO'}", (0, 255, 0) if speaking else (0, 0, 255))
        ]

        # 텍스트 출력 Loop
        curr_y = start_y
        for text, color in info_lines:
            # 배경 검은색 (가독성 확보)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(frame, (start_x - 5, curr_y - th - 5), (start_x + tw + 5, curr_y + 5), bg_color, -1)
            # 글자
            cv2.putText(frame, text, (start_x, curr_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
            curr_y += line_height

        # ---------------------------
        # 4️⃣ Confidence Bar (하단 배치)
        # ---------------------------
        conf_bar_x = start_x
        conf_bar_y = curr_y + 10
        bar_width = 150
        bar_height = 12
        
        # 배경 바
        cv2.rectangle(frame, (conf_bar_x, conf_bar_y), (conf_bar_x + bar_width, conf_bar_y + bar_height), (50, 50, 50), -1)
        # 게이지 바
        fill_width = int(conf * bar_width)
        cv2.rectangle(frame, (conf_bar_x, conf_bar_y), (conf_bar_x + fill_width, conf_bar_y + bar_height), (0, 255, 255), -1)
        # 텍스트
        conf_text = f"Conf: {conf:.2f}"
        cv2.putText(frame, conf_text, (conf_bar_x + bar_width + 10, conf_bar_y + bar_height - 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, (255, 255, 0), 1)

        # ---------------------------
        # 5️⃣ 우측 상단 상태 원 (신호등)
        # ---------------------------
        circle_x = w - 30
        circle_y = 30
        radius = 12

        if speaking:
            status_color = (0, 255, 0)    # Green (말함)
        elif person_detected:
            status_color = (0, 255, 255)  # Yellow (사람은 있음)
        else:
            status_color = (0, 0, 255)    # Red (아무도 없음)

        cv2.circle(frame, (circle_x, circle_y), radius, status_color, -1)
        cv2.circle(frame, (circle_x, circle_y), radius, (255, 255, 255), 1) # 테두리

        return frame