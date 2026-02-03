# 3D Printing Reverse Calibration Tool

Streamlit 기반의 3D 프린팅 역보정 및 검증 툴입니다.

## 배포 방법 (Streamlit Cloud)

이 프로젝트를 GitHub에 올린 후 Streamlit Cloud를 통해 무료로 웹에 배포할 수 있습니다.

### 1. GitHub 저장소 생성 및 코드 푸시
1. GitHub에 새 Repository를 생성합니다 (예: `3d-reverse-calibration`).
2. 로컬에서 Git 초기화 및 커밋:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
   git push -u origin main
   ```

### 2. Streamlit Cloud 연동
1. [Streamlit Cloud](https://streamlit.io/cloud)에 접속하여 GitHub 계정으로 로그인합니다.
2. **"New app"** 버튼을 클릭합니다.
3. 방금 생성한 Repository, Branch (`main`), Main file path (`app.py`)를 선택합니다.
4. **"Deploy!"** 버튼을 클릭합니다.

### 3. 완료
배포가 완료되면 전 세계 어디서든 접속 가능한 URL이 생성됩니다.

## 로컬 실행 방법
```bash
pip install -r requirements.txt
streamlit run app.py
```
