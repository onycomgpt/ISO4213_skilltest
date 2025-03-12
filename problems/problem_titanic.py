import streamlit as st
import pandas as pd

def display_problem_titanic():
    """문제 titanic를 화면에 표시하는 함수"""
    st.title("- 문제 3-1. Titanic (이진 분류) 성능검증")

    # 문제 설명
    st.markdown("""
    ## 📝 문제 설명
    <h2 style='font-size:28px;'>본 문항은 이진 분류 모델의 성능을 측정할 수 있는지 여부를 알아보는 문제입니다.</h2>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size: 22px; color: #555;">
    (단, 모든 결과는 소수점 세 번째 자리에서 반올림 할 것 - 두 번째 자리까지 표기)
    </div>
    """, unsafe_allow_html=True)

    # 2열 레이아웃 생성 (왼쪽: 시험 절차, 오른쪽: 오차 행렬)
    col1, col2 = st.columns([1, 1])  # 비율을 1:1로 설정, 필요 시 조정 가능

    # 왼쪽 열: 시험 절차
    with col1:
        st.markdown("""
        ## 📝 시험 절차
        <div style='border: 2px solid #333399; padding: 15px; border-radius: 10px; background-color: #F9F9FF;'>
            <h3 style='font-size:24px; color: #333399;'>1. 모델 및 데이터 다운로드</h3>
            <p style='font-size:20px;'>좌측 네비게이션바에서 [Titanic 모델 다운로드] 및 [Titanic 테스트 데이터 다운로드] 버튼을 클릭하여 다운로드하세요.</p>
            <br/><br/>
            <h3 style='font-size:24px; color: #333399;'>2. 지표 계산</h3>
            <p style='font-size:20px;'>주어진 모델과 데이터를 기반으로 성능지표에 대한 값을 도출하시오.</p>
            <p style='font-size:20px;'>(우측 오차 행렬은 참고용입니다.)</p>
        </div>
        """, unsafe_allow_html=True)

    # 오른쪽 열: 오차 행렬
    with col2:
        # st.markdown("<h2 style='text-align: left; font-size:28px; color: #333399;'>오차 행렬</h2>", unsafe_allow_html=True)
        image_path = "titanic_confusion_matrix.png"  # 이미지 경로 확인
        st.image(image_path, caption="Iris Confusion Matrix", width=600)  # 너비 조정

    # 문제 섹션 (전체 너비로 유지)
    st.markdown("---")  # 구분선 추가
    problem_list = [
        ("정확도 (Accuracy)", "#FF4B4B"),
        ("정밀도 (Precision)", "#FF4B4B"),
        ("재현율 (Recall)", "#FF4B4B"),
        ("F1-스코어 (F1-score)", "#FF4B4B"),
    ]

    for idx, (title, color) in enumerate(problem_list, start=1):
        st.markdown(f"""
        ## - 문제3-1-{idx}.
        <div style='border: 2px solid {color}; padding: 20px; border-radius: 12px; background-color: #FFF8F8;'>
            <h2 style='font-size:28px; color: {color};'>{title}를 구하시오.</h2>
        </div>
        <br>
        """, unsafe_allow_html=True)  # 문제 사이 간격 추가
