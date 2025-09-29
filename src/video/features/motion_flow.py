import cv2
import numpy as np


def optical_flow_feature(prev_gray, gray, prev_pts):
    if prev_gray is None or gray is None or prev_pts is None:
        return 0.0, None
    prev_pts = prev_pts.astype(np.float32).reshape(-1, 1, 2)
    cur, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
    if cur is None:
        return 0.0, None
    good = st.reshape(-1) == 1
    if not np.any(good):
        return 0.0, None
    prev_good = prev_pts.reshape(-1, 2)[good]
    cur_good = cur.reshape(-1, 2)[good]
    mags = np.linalg.norm(cur_good - prev_good, axis=1)
    
    return float(np.mean(mags)), cur.reshape(-1, 2)
