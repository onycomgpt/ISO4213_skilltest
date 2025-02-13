import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def preprocess_iris_data(test_df):
    """Iris 데이터 전처리"""
    if test_df is not None:
        try:
            X_test = test_df.drop(columns=['label'])
            y_test = test_df['label']

            scaler = StandardScaler()
            X_test_scaled = scaler.fit_transform(X_test)

            return X_test_scaled, y_test
        except Exception as e:
            print(f"❌ Iris 데이터 전처리 실패: {e}")
    return None, None

def calculate_iris_metrics(y_test, y_pred):
    """Iris 모델 성능 평가"""
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    # ✅ 매크로 평균 (모든 클래스 동일 가중치 평균)
    precision_macro = precision_score(y_test, y_pred, average="macro")
    recall_macro = recall_score(y_test, y_pred, average="macro")
    f1_macro = f1_score(y_test, y_pred, average="macro")

    # ✅ 가중 평균 (클래스 샘플 수에 따른 가중치 적용)
    precision_weighted = precision_score(y_test, y_pred, average="weighted")
    recall_weighted = recall_score(y_test, y_pred, average="weighted")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    # ✅ 마이크로 평균 (전체 샘플을 하나의 단일 집합으로 취급)
    precision_micro = precision_score(y_test, y_pred, average="micro")
    recall_micro = recall_score(y_test, y_pred, average="micro")
    f1_micro = f1_score(y_test, y_pred, average="micro")

    # ✅ Fβ-score 계산 (β = 0.3, Precision을 더 중요하게 고려)
    beta = 0.3
    f_beta = (1 + beta ** 2) * (precision_weighted * recall_weighted) / ((beta ** 2 * precision_weighted) + recall_weighted)

    # ✅ 최종 성능 지표 딕셔너리
    metrics = {
        "Accuracy": accuracy,

        "Precision (Macro)": precision_macro,
        "Recall (Macro)": recall_macro,
        "F1-score (Macro)": f1_macro,
        
        "Precision (Weighted)": precision_weighted,
        "Recall (Weighted)": recall_weighted,
        "F1-score (Weighted)": f1_weighted,
        
        "Precision (Micro)": precision_micro,
        "Recall (Micro)": recall_micro,
        "F1-score (Micro)": f1_micro,
        
        "Fβ-score (β=0.3)": f_beta
    }

    return metrics
