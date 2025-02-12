import streamlit as st
import pandas as pd

def display_problem2():
    """문제 2를 화면에 표시하는 함수"""
    st.title("📢 문제 2: 성능 지표 계산")

    # 문제 설명
    st.markdown("""
    ## 📝 문제 설명
    <h2 style='font-size:32px;'> 아래 오차행렬을 기반으로 다음의 성능 지표를 계산하시오.</h2>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="font-size: 26px;">
    (단, 모든 결과는 소수점 세 번째 자리에서 반올림 할 것 (두 번째 자리까지 표기))
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
    # 오차행렬 테이블
        st.markdown("<h2 style='text-align: center; font-size:36px;'>🔢 오차 행렬</h2>", unsafe_allow_html=True)

        # HTML 테이블을 활용하여 물리적 크기 확대
        st.markdown("""
        <table style="width:100%; height:400px; border-collapse: collapse; text-align: center; font-size: 36px; border: 3px solid #333399;">
            <tr style="background-color: #E3E4E6; font-size: 40px;">
                <th></th>
                <th>실제 양성 (Positive)</th>
                <th>실제 음성 (Negative)</th>
            </tr>
            <tr>
                <td style="font-weight: bold;">예측 양성 (Positive)</td>
                <td style="padding: 20px; border: 3px solid #333399;">70</td>
                <td style="padding: 20px; border: 3px solid #333399;">30</td>
            </tr>
            <tr>
                <td style="font-weight: bold;">예측 음성 (Negative)</td>
                <td style="padding: 20px; border: 3px solid #333399;">25</td>
                <td style="padding: 20px; border: 3px solid #333399;">175</td>
            </tr>
        </table>
        """, unsafe_allow_html=True)

    with col2:
        # TP, FP, FN, TN 정보 제공
        st.markdown("<h2 style='font-size:40px;text-align: center;'>📊 TP, FP, FN, TN 값</h2>", unsafe_allow_html=True)
        st.markdown("""
        <div style='font-size:34px; line-height:2.5; background-color:#F5F5F5; padding:30px; border-radius:12px; border: 3px solid #333399;'>
        ✅ TP (True Positive) = 70  <br>
        ✅ FP (False Positive) = 30  <br>
        ✅ FN (False Negative) = 25  <br>
        ✅ TN (True Negative) = 175  <br>
        </div>
    """, unsafe_allow_html=True)



    # 문제 섹션
    problem_list = [
        ("정확도 (Accuracy)", "#FF3333"),
        ("정밀도 (Precision)", "#FF8800"),
        ("재현율 (Recall)", "#33AA33"),
        ("특이도 (Specificity)", "#3377FF"),
        ("F1-스코어 (F1-score)", "#9900CC"),
        ("Fβ-스코어 (β=2)", "#CC0066")
    ]

    for idx, (title, color) in enumerate(problem_list, start=1):
        st.markdown(f"""
        ## 📝 문제2-{idx}.
        <div style='border: 4px solid {color}; padding: 20px; border-radius: 12px; background-color: #FFF8F8;'>
            <h2 style='font-size:28px; color: {color};'>{title}를 구하시오.</h2>
        </div>
        <br>
        """, unsafe_allow_html=True)  # 문제 사이 간격 추가
