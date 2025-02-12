import streamlit as st
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 모듈 가져오기
from modules.titanic_handler import preprocess_titanic_data, calculate_titanic_metrics
from modules.iris_handler import preprocess_iris_data, calculate_iris_metrics
from modules.mnist_handler import preprocess_mnist_data, calculate_mnist_metrics
from problems.problem1 import display_problem1
from problems.problem2 import display_problem2

# 페이지 설정 (전체 너비 사용)
st.set_page_config(layout="wide")

st.title("📢 ISO4213 Skill Test: Classification Model")

# 세션 상태 초기화
if "page" not in st.session_state:
    st.session_state["page"] = "home"

if "dataset_type" not in st.session_state:
    st.session_state["dataset_type"] = None

if "model_file" not in st.session_state:
    st.session_state["model_file"] = None

if "test_data_file" not in st.session_state:
    st.session_state["test_data_file"] = None

# 🎯 혼동 행렬 시각화 함수
def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)

# 🎯 모델 검증 실행 함수
def start_validation():
    dataset_type = st.session_state["dataset_type"]
    model_file = st.session_state["model_file"]
    test_data_file = st.session_state["test_data_file"]

    if model_file is not None and test_data_file is not None:
        model = joblib.load(model_file)
        test_df = pd.read_csv(test_data_file)

        if dataset_type == "titanic":
            X_test, y_test = preprocess_titanic_data(test_df)
            y_pred = model.predict(X_test)
            metrics = calculate_titanic_metrics(y_test, y_pred)
        elif dataset_type == "iris":
            X_test, y_test = preprocess_iris_data(test_df)
            y_pred = model.predict(X_test)
            metrics = calculate_iris_metrics(y_test, y_pred)
        elif dataset_type == "mnist":
            X_test, y_test = preprocess_mnist_data(test_df)
            y_pred = model.predict(X_test)
            metrics = calculate_mnist_metrics(y_test, y_pred)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("모델 성능 평가 결과")
            for key, value in metrics.items():
                if value is not None:
                    st.markdown(f"<p style='font-size:30px; font-weight:bold;'>✅ {key}: {value:.4f}</p>", unsafe_allow_html=True)
        
        with col2:
            plot_confusion_matrix(y_test, y_pred)

# 🎯 홈 화면 (시험 시작 버튼 중앙 정렬)
def home_page():
    st.markdown("""<div style="text-align: center;">
        <h1>🚀 환영합니다!</h1>
        <p style="font-size: 24px;">본 시험은 AI 모델 평가 및 이론 문제 풀이로 구성되어 있습니다.</p>
        <hr>
        <h2>📌 시험 구성</h2>
        <p style="font-size: 24px;"> AI 모델 성능 평가 개념 문제 풀이</p>
        <p style="font-size: 24px;"> 업로드된 모델을 평가하고 성능 확인</p>
        <br>
        <p style="font-size: 18px;">아래 버튼을 눌러 시험을 시작하세요.</p>
    </div>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("시험 시작하기 🚀", use_container_width=True):
            st.session_state["page"] = "exam"
            st.rerun()

# 🎯 시험 페이지
def exam_page():
    st.title("ISO4213 Skill Test")
    
    task_type = st.sidebar.selectbox("시험 유형 선택", ["이론 문제", "모델 검증"])
    
    if task_type == "이론 문제":
        problem_type = st.sidebar.selectbox("이론 문제 선택", ["문제1", "문제2"])
        if problem_type == "문제1":
            display_problem1()
        elif problem_type == "문제2":
            display_problem2()
    
    elif task_type == "모델 검증":
        dataset_type = st.sidebar.selectbox("분류 유형 선택", ["Titanic (이진 분류)", "Iris (다중 클래스)", "MNIST (다중 레이블)"])
        dataset_mapping = {"Titanic (이진 분류)": "titanic", "Iris (다중 클래스)": "iris", "MNIST (다중 레이블)": "mnist"}
        st.session_state["dataset_type"] = dataset_mapping[dataset_type]

        st.session_state["model_file"] = st.sidebar.file_uploader("모델 파일 업로드 (.pkl)", type=["pkl"])
        st.session_state["test_data_file"] = st.sidebar.file_uploader("테스트 데이터 업로드 (.csv)", type=["csv"])

        if st.sidebar.button("모델 검증 시작"):
            start_validation()

# 메인 실행 함수
def main():
    if st.session_state["page"] == "home":
        home_page()
    elif st.session_state["page"] == "exam":
        exam_page()

if __name__ == "__main__":
    main()
