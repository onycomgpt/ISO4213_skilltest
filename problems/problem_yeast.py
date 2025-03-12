import streamlit as st
import pandas as pd

def display_problem_yeast():
    """문제 yeast를 화면에 표시하는 함수"""
    st.title("- 문제 3-3. yeast (다중 라벨 분류) 성능검증")

    # 문제 설명
    st.markdown("""
    ## 📝 문제 설명
    <h2 style='font-size:28px;'>본 문항은 다중 라벨 분류 모델의 성능을 측정할 수 있는지 여부를 알아보는 문제입니다.</h2>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size: 22px; color: #555;">
    (단, 모든 결과는 소수점 세 번째 자리에서 반올림 할 것 - 두 번째 자리까지 표기)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ## 📝 시험 절차
    <div style='border: 2px solid #333399; padding: 15px; border-radius: 10px; background-color: #F9F9FF;'>
        <h3 style='font-size:24px; color: #333399;'>1. 모델 및 데이터 다운로드</h3>
        <p style='font-size:20px;'>좌측 네비게이션바에서 [yeast 모델 다운로드] 및 [yeast 테스트 데이터 다운로드] 버튼을 클릭하여 다운로드하세요.</p>
        <br/><br/>
        <h3 style='font-size:24px; color: #333399;'>2. 지표 계산</h3>
        <p style='font-size:20px;'>주어진 모델과 데이터를 기반으로 성능지표에 대한 값을 도출하시오.</p>
        
    </div>
    """, unsafe_allow_html=True)

    # 문제 섹션 (전체 너비로 유지)
    st.markdown("---")  # 구분선 추가
    problem_list = [
        ("Hamming Loss", "#FF4B4B"),
        ("Exact Match Ratio", "#FF4B4B"),
        ("Jaccard Index", "#FF4B4B"),
        # ("KL Divergence", "#FF4B4B"),
    ]

    for idx, (title, color) in enumerate(problem_list, start=1):
        st.markdown(f"""
        ## - 문제 3-3-{idx}.
        <div style='border: 2px solid {color}; padding: 20px; border-radius: 12px; background-color: #FFF8F8;'>
            <h2 style='font-size:28px; color: {color};'>{title}를 구하시오.</h2>
        </div>
        <br>
        """, unsafe_allow_html=True)  # 문제 사이 간격 추가
