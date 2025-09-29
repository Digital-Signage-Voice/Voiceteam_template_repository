from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

# -------------------------------
# 예시 학습 데이터
# -------------------------------

# X_train : 입력(feature) 데이터
# 각 요소는 한 프레임(혹은 샘플)에서 추출한 특징 3가지
#   [lip_ratio, flow_mag, diff_val]
#   lip_ratio : 입술 비율 (입술 길이 대비 벌어진 정도)
#   flow_mag  : Optical Flow로 계산한 입술 움직임 세기
#   diff_val  : 이전 프레임과 현재 프레임 픽셀 차이 합계
X_train = [
    [0.3, 0.05, 0.01],  # 샘플1: 입술 비율 0.3, 움직임 0.05, 픽셀 차 0.01
    [0.4, 0.1, 0.02],   # 샘플2: 입술 비율 0.4, 움직임 0.1, 픽셀 차 0.02
    # 실제 학습 시에는 더 많은 프레임 데이터를 수집하여 넣어야 함
]

# y_train : 출력(label) 데이터
# 각 요소는 X_train의 각 샘플이 발화 중인지 아닌지에 대한 정답
#   0 : 비발화 (말하지 않음)
#   1 : 발화   (말하고 있음)
y_train = [
    0,  # 샘플1은 발화하지 않음
    1,  # 샘플2는 발화 중
    # 실제 학습 시에는 X_train과 길이가 동일해야 함
]


# 스케일러 학습
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# SVM 모델 학습
model = SVC(probability=True)
model.fit(X_scaled, y_train)

# 저장
joblib.dump(scaler, "classification/scaler.pkl")
joblib.dump(model, "classification/ml_model.pkl")
print("모델과 스케일러 저장 완료")
