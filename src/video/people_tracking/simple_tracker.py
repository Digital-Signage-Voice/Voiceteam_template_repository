import numpy as np

class SimpleTracker:
    """
    CPU 전용 단순 Kalman Filter 기반 사람 추적
    - ReID 없이 위치만 기반
    """
    def __init__(self):
        self.tracks = {}
        self.next_id = 0

    def update(self, detections):
        active_tracks = []
        for det in detections:
            track = {
                'track_id': self.next_id,
                'bbox': det['bbox'],
                'conf': det['conf']
            }
            self.tracks[self.next_id] = track
            self.next_id += 1
            active_tracks.append(track)
        return active_tracks
