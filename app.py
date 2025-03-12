import streamlit as st
import os
import pandas as pd
import joblib
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 모듈 가져오기
from modules.titanic_handler import preprocess_titanic_data, calculate_titanic_metrics
from modules.iris_handler import preprocess_iris_data, calculate_iris_metrics
from modules.yeast_handler import preprocess_yeast_data, calculate_yeast_metrics
from problems.problem1 import display_problem1
from problems.problem2 import display_problem2
from problems.problem_titanic import display_problem_titanic
from problems.problem_iris import display_problem_iris
from problems.problem_yeast import display_problem_yeast


# 페이지 설정 (전체 너비 사용)
st.set_page_config(layout="wide")

st.title("📢 ISO4213 Skill Test: Classification Model")

# 세션 상태 초기화
if "page" not in st.session_state:
    st.session_state["page"] = "home"

if "dataset_type" not in st.session_state:
    st.session_state["dataset_type"] = None

if "uploaded_model" not in st.session_state:
    st.session_state["uploaded_model"] = None

if "uploaded_test_data" not in st.session_state:
    st.session_state["uploaded_test_data"] = None

# 📂 assets 폴더 내의 데이터셋 및 모델 파일 자동 매핑
ASSETS_PATH = "assets"

DATASET_FILES = {
    "titanic": {
        "model": os.path.join(ASSETS_PATH, "titanic", "titanic_model.pkl"),
        "test_data": os.path.join(ASSETS_PATH, "titanic", "titanic_test_dataset.csv")
    },
    "iris": {
        "model": os.path.join(ASSETS_PATH, "iris", "iris_model.pkl"),
        "test_data": os.path.join(ASSETS_PATH, "iris", "iris_test_dataset.csv")
    },
    "yeast": {
    "model": os.path.join(ASSETS_PATH, "yeast", "yeast_model.pkl"),
    "test_data": os.path.join(ASSETS_PATH, "yeast", "yeast_test_dataset.csv")
    }
}


# 🎯 홈 화면 (랜딩 페이지)
def home_page():
    # 로고 추가 (좌측 상단)
    st.image("onycom_logo.png", width=150)  # 로고 크기 조정

    # 제목 추가 (중앙 정렬)
    st.markdown("<h1 style='text-align: center;'>🚀 환영합니다!</h1>", unsafe_allow_html=True)

    # 시험 설명 추가 (중앙 정렬)
    st.markdown("""
    <div style='text-align: center; font-size: 24px;'>
        본 시험은 <b>개념 문제 및 모델 검증</b>으로 구성되어 있습니다.<br>
        아래 버튼을 눌러 시험을 시작하세요.
    </div>
    <br><br>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("시험 시작하기 🚀", use_container_width=True):
            st.session_state["page"] = "exam"
            st.rerun()


# 🎯 업로드된 파일을 임시 저장하는 함수
def save_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return temp_path
    return None

# 🎯 혼동 행렬 시각화 함수
def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)


# 🎯 모델 검증 실행 함수
# 🎯 모델 검증 실행 함수 (dataset_type과 파일 이름 검사 추가)
def start_validation():
    dataset_type = st.session_state["dataset_type"]

    # 업로드된 파일 확인
    if not st.session_state["uploaded_model"]:
        st.warning("⚠️ 모델을 업로드하세요.")
        return  
    if not st.session_state["uploaded_test_data"]:
        st.warning("⚠️ 테스트 데이터를 업로드하세요.")
        return  

    # 업로드된 파일을 임시 저장 후 사용
    uploaded_model = save_uploaded_file(st.session_state["uploaded_model"])
    uploaded_test_data = save_uploaded_file(st.session_state["uploaded_test_data"])

    # 업로드된 파일이 없으면 실행하지 않음
    if not uploaded_model or not uploaded_test_data:
        return  

    # 🔹 업로드된 파일명 검사 (dataset_type과 비교)
    # 기대하는 파일명 가져오기
    expected_model_name = os.path.basename(DATASET_FILES[dataset_type]["model"])
    expected_test_data_name = os.path.basename(DATASET_FILES[dataset_type]["test_data"])

    # 업로드된 파일명 가져오기
    uploaded_model_name = st.session_state["uploaded_model"].name
    uploaded_test_data_name = st.session_state["uploaded_test_data"].name

    # 파일명이 일치하는지 확인
    if uploaded_model_name != expected_model_name or uploaded_test_data_name != expected_test_data_name:
        st.error(f"❌ 업로드된 모델과 테스트 데이터가 {dataset_type} 유형과 일치하지 않습니다(파일명 확인: XXX_model / XXX_test_dataset).")
        return


    # 🔹 모델과 데이터셋 로드
    model = joblib.load(uploaded_model)
    test_df = pd.read_csv(uploaded_test_data)

    # 🔹 데이터셋 유형에 따른 처리
    if dataset_type == "titanic":
        X_test, y_test = preprocess_titanic_data(test_df)
        y_pred = model.predict(X_test)
        metrics = calculate_titanic_metrics(y_test, y_pred)
    elif dataset_type == "iris":
        X_test, y_test = preprocess_iris_data(test_df)
        y_pred = model.predict(X_test)
        metrics = calculate_iris_metrics(y_test, y_pred)
    elif dataset_type == "yeast":
        X_test, y_test = preprocess_yeast_data(test_df)
        y_pred = model.predict(X_test)
        metrics = calculate_yeast_metrics(y_test, y_pred)

    # 🔹 성능 결과 출력
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("모델 성능 평가 결과")
        for key, value in metrics.items():
            if value is not None:
                st.markdown(f"<p style='font-size:30px; font-weight:bold;'>✅ {key}: {value:.4f}</p>", unsafe_allow_html=True)

    with col2:
        # 다중 라벨인지 확인 (2D 배열인지 체크)
        is_multilabel = len(y_test.shape) > 1

        if is_multilabel:
            st.markdown("<h3 style='text-align: center;'>📊 다중 라벨 성능 지표 그래프</h3>", unsafe_allow_html=True)
            plot_multilabel_metrics(metrics)  # 다중 라벨 → 막대 그래프 출력
        else:
            st.markdown("<h3 style='text-align: center;'>🟦 Confusion Matrix</h3>", unsafe_allow_html=True)
            plot_confusion_matrix(y_test, y_pred)  # 다중 클래스 → Confusion Matrix 출력


# 🎯 모델 및 데이터 다운로드 버튼 추가
def add_download_buttons(dataset_type):
    """선택된 데이터셋에 맞는 모델과 테스트 데이터 다운로드 버튼을 추가"""
    if dataset_type in DATASET_FILES:
        model_path = DATASET_FILES[dataset_type]["model"]
        test_data_path = DATASET_FILES[dataset_type]["test_data"]

        # 모델 다운로드 버튼 추가
        if os.path.exists(model_path):
            with open(model_path, "rb") as file:
                st.sidebar.download_button(
                    label=f"📥 {dataset_type.capitalize()} 모델 다운로드",
                    data=file,
                    file_name=f"{dataset_type}_model.pkl",
                    mime="application/octet-stream"
                )

        # 테스트 데이터 다운로드 버튼 추가
        if os.path.exists(test_data_path):
            with open(test_data_path, "rb") as file:
                st.sidebar.download_button(
                    label=f"📥 {dataset_type.capitalize()} 테스트 데이터 다운로드",
                    data=file,
                    file_name=f"{dataset_type}_test_dataset.csv",
                    mime="text/csv"
                )

# 🎯 시험 페이지
def exam_page():
    # st.title("ISO4213 Skill Test")
    
    task_type = st.sidebar.selectbox("시험 유형 선택", ["개념 문제", "모델 검증"])
    
    if task_type == "개념 문제":
        problem_type = st.sidebar.selectbox("개념 문제 선택", ["이론 문제", "성능 지표 계산"])
        if problem_type == "이론 문제":
            display_problem1()
        elif problem_type == "성능 지표 계산":
            display_problem2()
    
    elif task_type == "모델 검증":
        dataset_type = st.sidebar.selectbox("분류 유형 선택", ["Titanic (이진 분류)", "Iris (다중 클래스)", "Yeast (다중 라벨)"])
        dataset_mapping = {"Titanic (이진 분류)": "titanic", "Iris (다중 클래스)": "iris", "Yeast (다중 라벨)": "yeast"}
        st.session_state["dataset_type"] = dataset_mapping[dataset_type]

        if dataset_type == "Titanic (이진 분류)":
            display_problem_titanic()
        if dataset_type == "Iris (다중 클래스)":
            display_problem_iris()
        if dataset_type == "Yeast (다중 라벨)":
            display_problem_yeast()

        # 모델 및 데이터 다운로드 버튼 추가
        add_download_buttons(st.session_state["dataset_type"])

        # # 모델 및 데이터 업로드 기능 추가
        # st.session_state["uploaded_model"] = st.sidebar.file_uploader("모델 업로드 (.pkl)", type=["pkl"])
        # st.session_state["uploaded_test_data"] = st.sidebar.file_uploader("테스트 데이터 업로드 (.csv)", type=["csv"])

        # if st.sidebar.button("모델 검증 시작"):
        #     start_validation()

def plot_multilabel_metrics(metrics):
    """다중 라벨 성능 지표 시각화"""
    
    # 성능 지표 목록 및 값
    metric_names = ["Hamming Loss", "Exact Match Ratio", "Jaccard Index", "KL Divergence"]
    metric_values = [metrics.get(name, 0) for name in metric_names]

    # ✅ 막대 그래프 생성
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(metric_names, metric_values, color=["red", "green", "blue", "purple"])
    ax.set_xlabel("Metric Value")
    ax.set_title("Multi-label Classification Metrics")
    
    # ✅ Streamlit에 그래프 출력
    st.pyplot(fig)

# 🎯 메인 실행 함수 (홈 페이지 & 시험 페이지 연결)
def main():
    if "page" not in st.session_state:
        st.session_state["page"] = "home"

    if st.session_state["page"] == "home":
        home_page()
    elif st.session_state["page"] == "exam":
        exam_page()

# 앱 실행
if __name__ == "__main__":
    main()