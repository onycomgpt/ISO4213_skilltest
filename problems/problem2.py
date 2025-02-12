import streamlit as st
import pandas as pd

def display_problem2():
    """문제 2를 화면에 표시하는 함수"""
    st.title("📢 문제 2: 성능 지표 계산")

    # 문제 설명
    st.markdown("""
    ### 📝 문제 설명
    <h2 style='font-size:30px;'> 아래 오차행렬을 기반으로 다음의 성능 지표를 계산하시오.</h2>
    """,unsafe_allow_html=True)
    
    st.markdown("""
    **(단, 모든 결과는 소수점 세 번째 자리에서 반올림 할 것(두번째 자리까지 표기))**
    """)

    # 오차행렬 테이블
    st.markdown("### 🔢 오차 행렬")
    data = {
        "실제 양성 (Positive)": [70, 25],
        "실제 음성 (Negative)": [30, 175]
    }
    df = pd.DataFrame(data, index=["예측 양성 (Positive)", "예측 음성 (Negative)"])
    st.dataframe(df.style.set_properties(**{'text-align': 'center', 'font-size': '16px'}).set_table_styles(
        [{'selector': 'th', 'props': [('font-size', '18px'), ('text-align', 'center')]}]
    ))

    col1, col2 = st.columns(2)
    with col1:
        # TP, FP, FN, TN 정보 제공
        st.markdown("### 📊 TP, FP, FN, TN 값")
        st.markdown("""
        - **TP (True Positive) = 70**
        - **FP (False Positive) = 30**
        - **FN (False Negative) = 25**
        - **TN (True Negative) = 175**
        """)
    with col2:
        # 성능 지표 목록
        st.markdown("### 📌 계산할 성능 지표")
        st.markdown("""
        - **1. 정확도 (Accuracy)**
        - **2. 정밀도 (Precision)**
        - **3. 재현율 (Recall)**
        - **4. 특이도 (Specificity)**
        - **5. F1-스코어 (F1-score)**
        - **6. Fβ-스코어 (β=2)**
        """)
