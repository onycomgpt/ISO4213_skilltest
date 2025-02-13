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


from sklearn.metrics import hamming_loss, accuracy_score, jaccard_score

def calculate_mnist_metrics(y_test, y_pred):
    """MNIST 모델 성능 평가 (다중 라벨 방식 적용)"""
    
    # 🔹 해밍 손실 (Hamming Loss) - 낮을수록 좋음
    hamming = hamming_loss(y_test, y_pred)

    # 🔹 정확 일치 비율 (Exact Match Ratio, Subset Accuracy)
    exact_match = accuracy_score(y_test, y_pred)

    # 🔹 자카드 지수 (Jaccard Index) - 0~1 사이 값 (1에 가까울수록 좋음)
    jaccard = jaccard_score(y_test, y_pred, average="samples")  

    return {
        "Hamming Loss": hamming,       # 낮을수록 좋음
        "Exact Match Ratio": exact_match,  # 높을수록 좋음
        "Jaccard Index": jaccard       # 1에 가까울수록 좋음
    }
