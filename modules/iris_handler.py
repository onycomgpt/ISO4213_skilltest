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
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    
    # 다중 클래스 분류인지 확인
    if cm.shape == (2, 2):  # 이진 분류일 경우
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        tn = fp = fn = tp = specificity = None  # 다중 클래스 분류에서는 특이도 계산 불가
    
    beta = 0.3
    f_beta = (1 + beta ** 2) * (precision * recall) / ((beta ** 2 * precision) + recall)
    
    metrics = {
        "Accuracy": accuracy, "Precision": precision,
        "Recall": recall, "F1-score": f1, "Fβ-score": f_beta
    }

    # 이진 분류일 경우 추가 지표 포함
    if tn is not None:
        metrics.update({
            "TP": tp, "TN": tn, "FP": fp, "FN": fn, "Specificity": specificity
        })

    return metrics