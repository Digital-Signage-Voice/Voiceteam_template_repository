import os
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

# -------------------------------
# 모델 저장 폴더 및 파일 경로
# -------------------------------
BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "classification")
os.makedirs(BASE_DIR, exist_ok=True)  

SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "ml_model.pkl")

# -------------------------------
# 예시 학습 데이터
# -------------------------------

# X_train : 입력(feature) 데이터
# 각 요소는 한 프레임(혹은 샘플)에서 추출한 특징 3가지
# [lip_radio, diff_val] = [입술 길이 대비 벌어진 정도, 이전 프레임과 현재 프레임 픽셀 차이 합계]

X_train = [
    [0.3, 0.01],
    [0.4, 0.02],
    [0.35, 0.015],
    # 실제 데이터는 수천~만 프레임 이상 필요
]

# y_train : 출력(label) 데이터
# 각 요소는 X_train의 각 샘플이 발화 중인지 아닌지에 대한 정답
#   0 : 비발화 (말하지 않음)
#   1 : 발화   (말하고 있음)
# x_train과 길이가 동일해야 함

y_train = [
    0, 
    1,
    0
]


# 스케일러 학습
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# SVM 모델 학습
model = SVC(probability=True)
model.fit(X_scaled, y_train)

# 모델 저장
joblib.dump(scaler, SCALER_PATH)
joblib.dump(model, MODEL_PATH)
print("Model and scaler saved successfully")
