import cv2
import numpy as np


class Overlay:
    @staticmethod
    def draw(frame, lip_pts, ratio, flow_mag, diff_val, speaking):
        if lip_pts is not None and len(lip_pts) > 0:
            for p in lip_pts:
                cv2.circle(frame, tuple(p.astype(int)), 2, (0, 255, 0), -1)
            x, y, w, h = cv2.boundingRect(lip_pts.astype(np.int32))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
        txt = f"L_R: {ratio:.3f} FM: {flow_mag:.3f} FD: {diff_val:.3f}"
        cv2.putText(frame, txt, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        status = "SPEAKING" if speaking else "silent"
        color = (0, 255, 0) if speaking else (0, 0, 255)
        cv2.putText(frame, status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
        return frame
