from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_classification(y_test, y_pred, dataset_type):
    """이진 분류, 다중 클래스 분류, 다중 레이블 분류 성능 평가"""
    if dataset_type == "titanic":
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        return {
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "Accuracy": accuracy, "Precision": precision,
            "Recall": recall, "Specificity": specificity,
            "F1-score": f1
        }

    elif dataset_type == "zoo":
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")
        return {
            "Accuracy": accuracy, "Precision": precision,
            "Recall": recall, "F1-score": f1
        }

def plot_confusion_matrix(y_test, y_pred):
    """혼동 행렬 시각화"""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    return fig
