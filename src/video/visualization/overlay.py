import cv2
import numpy as np


class Overlay:
    @staticmethod
    def draw(frame, result):
        """
        frame: BGR 이미지
        result: processor.py에서 반환된 dict
        """
        # ROI 그리기
        roi = result.get("roi")
        if roi:
            x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 발화 상태 표시
        speaking = result.get("is_speaking", False)
        status = "SPEAKING" if speaking else "silent"
        color = (0, 255, 0) if speaking else (0, 0, 255)
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # 발화 구간 수 표시
        speech_count = len(result.get("speech_segments", []))
        cv2.putText(frame, f"Speech Segments: {speech_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        # flags 표시 (간단)
        flags = result.get("flags", {})
        flag_txt = f"ROI:{flags.get('roi_detected',False)} TS:{flags.get('timestamp_valid',False)}"
        cv2.putText(frame, flag_txt, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 추가: L_R, FD 값은 필요 시 result dict에 포함시켜서 표시 가능
        return frame
