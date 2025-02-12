import os
import sys

# 실행할 포트 설정
PORT = 8501

# PyInstaller 실행 파일이면 `_MEIPASS` 임시 폴더에서 실행
if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS  # PyInstaller 임시 디렉터리
    python_cmd = os.path.join(base_path, "python.exe")  # `python.exe` 실행
else:
    base_path = os.path.dirname(os.path.abspath(__file__))
    python_cmd = sys.executable  # 개발 환경에서는 현재 Python 사용

app_path = os.path.join(base_path, "app.py")  # 압축 해제된 폴더에서 `app.py` 찾기

# Python 실행 파일이 없으면 오류 출력 후 종료
if not os.path.exists(python_cmd):
    print(f"⚠ 실행할 Python 실행 파일을 찾을 수 없습니다: {python_cmd}")
    input("Press Enter to exit...")
    sys.exit(1)

# `app.py`가 없으면 오류 출력 후 종료
if not os.path.exists(app_path):
    print(f"⚠ 실행할 앱 파일을 찾을 수 없습니다: {app_path}")
    input("Press Enter to exit...")
    sys.exit(1)

# 환경 변수 설정 (Python 표준 라이브러리 경로 포함)
os.environ["PYTHONHOME"] = base_path
os.environ["PYTHONPATH"] = os.path.join(base_path, "Lib") + ";" + os.path.join(base_path, "DLLs")

# Streamlit 실행 명령어
command = f'"{python_cmd}" -m streamlit run "{app_path}" --server.port {PORT} --server.headless true'
print(f"Running command: {command}")

# Streamlit 실행
os.system(command)

# 창이 바로 닫히지 않도록 대기
input("Press Enter to exit...")
