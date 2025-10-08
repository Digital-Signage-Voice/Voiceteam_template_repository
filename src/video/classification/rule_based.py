from config import cfg


class RuleBasedClassifier:
    def __init__(self):
        # 입술 연속 개방 프레임 카운터 초기화
        self.open_counter = 0


    def predict(self, lip_ratio, frame_diff_val):
        # 입술 정보가 없으면 카운터 초기화 후 비발화
        if lip_ratio is None:
            self.open_counter = 0
            return False
        
        # 입술이 열렸다고 판단되는 경우 카운터 증가
        if lip_ratio >= cfg.lip_ratio_threshold:
            self.open_counter += 1
        else:
            # 연속 개방 조건이 깨지면 카운터 초기화
            self.open_counter = 0
            
        # 연속 개방 프레임이 충분히 쌓이면 추가 조건 검사
        if self.open_counter >= cfg.open_frames_to_speaking:
            # 프레임 변화가 일정 이상이면 발화로 판단
            return frame_diff_val > 0.02
    
        return False
