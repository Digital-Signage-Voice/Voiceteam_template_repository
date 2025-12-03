from config import cfg
from collections import deque
import numpy as np

class RuleBasedClassifier:
    def __init__(self):
        # 최근 프레임의 입 상태 및 frame_diff 기록 (window)
        self.lip_states = deque(maxlen=6)  # 최근 6프레임 저장
        self.diff_vals = deque(maxlen=6)

    def predict(self, lip_ratio, frame_diff_val):
        """
        입 벌림 여부와 프레임 변화량을 종합하여 발화 여부(True/False) 판단
        """
        # 데이터가 없으면 초기화
        if lip_ratio is None:
            self.lip_states.clear()
            self.diff_vals.clear()
            return False

        # 1. 현재 프레임 입 열림 여부 (Threshold: 0.23 등 Config 설정값 따름)
        is_open = int(lip_ratio >= cfg.lip_ratio_threshold)
        
        # 2. 상태 저장
        self.lip_states.append(is_open)
        self.diff_vals.append(frame_diff_val)
        
        # 데이터가 충분히 모이지 않았으면 일단 False
        if len(self.lip_states) < 3:
            return False
      
        # [조건 1] 즉시 판정 (Fast Trigger)
        if is_open == 1 and frame_diff_val > 0.02:
            return True

        # [조건 2] 관성 판정 (Sustained Speaking)
        recent_open_count = list(self.lip_states).count(1)
        max_recent_diff = max(self.diff_vals)
        
        if recent_open_count >= 2 and max_recent_diff > 0.025:
            return True

        # [조건 3] 패턴 감지 (Flicker)
        states = list(self.lip_states)
        if len(states) >= 3:
            if (states[-1] != states[-2]) and max_recent_diff > 0.025:
                return True

        return False