import joblib
import os

class MLClassifier:
    def __init__(self):
        # 모델과 스케일러 파일 경로
        scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")
        model_path = os.path.join(os.path.dirname(__file__), "ml_model.pkl")

        # 학습된 스케일러와 모델 로드
        if os.path.exists(scaler_path) and os.path.exists(model_path):
            self.scaler = joblib.load(scaler_path)
            self.model = joblib.load(model_path)
        else:
            raise FileNotFoundError("Scaler or model file not found. 학습 후 저장된 파일이 필요합니다.")

    def predict(self, feat):
        """
        feat: [lip_ratio, flow_mag, diff_val] 리스트
        반환: True -> 발화, False -> 비발화
        """
        
        # 입력 특성 스케일링
        feat_scaled = self.scaler.transform([feat])
        # 예측 확률
        prob = self.model.predict_proba(feat_scaled)[0]
        
        # 확률 기준 0.5 이상이면 speaking
        return prob[1] > 0.5
