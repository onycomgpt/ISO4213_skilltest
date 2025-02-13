import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import hamming_loss, accuracy_score, jaccard_score
from scipy.stats import entropy
# from scipy.spatial.distance import wasserstein_distance
from scipy.stats import wasserstein_distance  # ✅ 올바른 임포트 경로

import numpy as np

def preprocess_zoo_data(test_df):
    """Zoo 데이터 전처리"""
    if test_df is not None:
        try:
            X_test = test_df.drop(columns=[col for col in test_df.columns if col.startswith('label_')])
            y_test = test_df[[col for col in test_df.columns if col.startswith('label_')]]

            scaler = StandardScaler()
            X_test_scaled = scaler.fit_transform(X_test)

            return X_test_scaled, y_test
        except Exception as e:
            print(f"❌ Zoo 데이터 전처리 실패: {e}")
    return None, None

def calculate_distribution_distance(y_test, y_pred):
    """분포 차이 (KL Divergence)"""
    y_test_dist = np.mean(y_test, axis=0)
    y_pred_dist = np.mean(y_pred, axis=0)
    
    # KL Divergence 계산 (로그 0 방지 위해 epsilon 추가)
    epsilon = 1e-10
    kl_div = entropy(y_test_dist + epsilon, y_pred_dist + epsilon)
        
    return kl_div

def calculate_zoo_metrics(y_test, y_pred):
    """Zoo 모델 성능 평가 (다중 라벨 분류)"""
    accuracy = accuracy_score(y_test, y_pred)

    # ✅ 해밍 손실 (Hamming Loss) - 낮을수록 좋음
    hamming = hamming_loss(y_test, y_pred)

    # ✅ 정확 일치 비율 (Exact Match Ratio, Subset Accuracy) - 높을수록 좋음
    exact_match = accuracy_score(y_test, y_pred)

    # ✅ 자카드 지수 (Jaccard Index) - 0~1 사이 값 (1에 가까울수록 좋음)
    jaccard = jaccard_score(y_test, y_pred, average="samples")

    # ✅ 분포 차이 메트릭 (KL Divergence & Wasserstein Distance)
    kl_div = calculate_distribution_distance(y_test, y_pred)

    # ✅ 최종 성능 지표 딕셔너리
    metrics = {
        "Hamming Loss": hamming,
        "Exact Match Ratio": exact_match,
        "Jaccard Index": jaccard,
        "KL Divergence": kl_div
    }
    
    return metrics
