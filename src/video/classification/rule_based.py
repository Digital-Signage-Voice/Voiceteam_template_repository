from config import cfg


class RuleBasedClassifier:
    def __init__(self):
        self.open_counter = 0


    def predict(self, lip_ratio, flow_mag, frame_diff_val):
        if lip_ratio is None:
            self.open_counter = 0
            return False
        if lip_ratio >= cfg.lip_ratio_threshold:
            self.open_counter += 1
        else:
            self.open_counter = 0
        if self.open_counter >= cfg.open_frames_to_speaking:
            if flow_mag is None:
                return True
            return flow_mag >= cfg.flow_magnitude_threshold or frame_diff_val > 0.02
    
        return False
