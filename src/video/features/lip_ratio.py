import numpy as np


def lip_aspect_ratio(lip_points):
    if lip_points is None or len(lip_points) < 6:
        return None
    x_coords = lip_points[:, 0]
    y_coords = lip_points[:, 1]
    left = lip_points[np.argmin(x_coords)]
    right = lip_points[np.argmax(x_coords)]
    top = lip_points[np.argmin(y_coords)]
    bottom = lip_points[np.argmax(y_coords)]
    horiz = np.linalg.norm(right - left) + 1e-6
    vert = np.linalg.norm(bottom - top) + 1e-6
    
    return float(vert / horiz)
