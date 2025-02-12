import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def preprocess_titanic_data(test_df):
    """Titanic 데이터 전처리 (이미 처리된 데이터는 추가 변환하지 않음)"""
    if test_df is not None:
        try:
            # 불필요한 컬럼 제거 (이전 모델 학습 시 제거한 컬럼과 일치해야 함)
            drop_columns = ['name', 'ticket', 'cabin', 'boat', 'body', 'home.dest']
            test_df = test_df.drop(columns=[col for col in drop_columns if col in test_df.columns], errors="ignore")

            # 범주형 데이터 처리 (이미 처리된 경우 생략)
            if 'sex' in test_df.columns or 'embarked' in test_df.columns:
                test_df = pd.get_dummies(test_df, columns=['sex', 'embarked'], drop_first=True)

            # 특성과 라벨 분리
            X_test = test_df.drop(columns=['label'])
            y_test = test_df['label']

            # 데이터 표준화 (훈련된 모델과 동일한 방식으로 변환)
            scaler = StandardScaler()
            X_test_scaled = scaler.fit_transform(X_test)

            return X_test_scaled, y_test
        except Exception as e:
            print(f"❌ Titanic 데이터 전처리 실패: {e}")
    return None, None

def calculate_titanic_metrics(y_test, y_pred):
    """Titanic 모델 성능 평가"""
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "Accuracy": accuracy, "Precision": precision,
        "Recall": recall, "F1-score": f1
    }
