from config import cfg
from collections import deque


class RuleBasedClassifier:
    def __init__(self):
        # 최근 프레임의 입 상태 및 frame_diff 기록 (window)
        self.lip_states = deque(maxlen=6)  # 최근 6프레임 저장
        self.diff_vals = deque(maxlen=6)


    def predict(self, lip_ratio, frame_diff_val):
        if lip_ratio is None:
            self.lip_states.clear()
            self.diff_vals.clear()
            return False
        is_open = int(lip_ratio >= cfg.lip_ratio_threshold)
        self.lip_states.append(is_open)
        self.diff_vals.append(frame_diff_val)
        # 입이 열렸을 때만 frame_diff, 패턴 체크
        if is_open == 1:
            states = list(self.lip_states)
            diffs = list(self.diff_vals)
            for i in range(1, len(states)-1):
                if states[i-1] == 1 and states[i] == 0 and states[i+1] == 1:
                    if max(diffs[i-1:i+2]) > 0.03:
                        return True
            if states.count(1) >= 4 and max(diffs) > 0.04:
                return True
        return False
