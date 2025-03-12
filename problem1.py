import streamlit as st

def display_problem1():
    """문제 1를 화면에 표시하는 함수"""
    st.title("- 문제 1: 이론 문제")

    # 문제 설명
    st.markdown("""
    ## 📝 문제1-1.
    <div style='border: 2px solid #FF4B4B; padding: 15px; border-radius: 10px;'>
        <h2 style='font-size:24px;'>특정 분류모델의 성능을 평가하는 지표로,<br><br> 실제값과 모델이 예측한 예측값을 
        한 눈에 알아 볼 수 있게 배열한 행렬은 [ ] 이다.</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""<br><br>""", unsafe_allow_html=True)  # 문제 사이 간격 추가

    st.markdown("""
    ## 📝 문제1-2.
    <div style='border: 2px solid #FF4B4B; padding: 15px; border-radius: 10px;'>
        <h2 style='font-size:24px;'>정밀도와 재현율의 조화평균으로 구해지는 평가지표는 [ ] 이다.</h2>
    </div>
    """, unsafe_allow_html=True)