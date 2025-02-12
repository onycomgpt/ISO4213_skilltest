import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def preprocess_mnist_data(test_df):
    """MNIST 데이터 전처리"""
    if test_df is not None:
        try:
            X_test = test_df.drop(columns=['label']).to_numpy()  # NumPy 배열 변환
            y_test = test_df['label'].to_numpy()  # NumPy 배열 변환
            X_test = X_test.astype(np.float32) / 255.0  # Normalize & float 변환

            return X_test, y_test
        except Exception as e:
            print(f"❌ MNIST 데이터 전처리 실패: {e}")
    return None, None

def calculate_mnist_metrics(y_test, y_pred):
    """MNIST 모델 성능 평가"""
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    return {
        "Accuracy": accuracy, "Precision": precision,
        "Recall": recall, "F1-score": f1
    }
