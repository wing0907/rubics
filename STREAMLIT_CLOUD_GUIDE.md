# 📤 Streamlit Cloud 배포 완벽 가이드

## 1단계: GitHub 저장소 생성

### A. GitHub 계정 로그인
- https://github.com 접속
- 우상단 `+` 아이콘 → `New repository` 클릭

### B. 저장소 설정
- **Repository name**: `qube-mvp` (또는 원하는 이름)
- **Description**: `Qube Mobile Learning MVP`
- **Public** 선택 (공개)
- **Add a README file** 체크 해제 (나중에 푸시할 예정)
- **Create repository** 클릭

### C. GitHub 저장소 URL 복사
생성 후 화면에 표시되는 URL을 복사합니다. 예:
```
https://github.com/[당신의username]/qube-mvp.git
```

---

## 2단계: 로컬에서 GitHub로 푸시

### PowerShell에서 실행:

```powershell
# 1. 저장소 디렉토리 이동
cd D:\Users\장우진\dev26

# 2. Git 초기화
git init
git config user.name "Your Name"
git config user.email "your.email@example.com"

# 3. 모든 파일 추가 (secrets.toml, .env 제외 - .gitignore 자동)
git add .

# 4. 첫 번째 커밋
git commit -m "Initial MVP commit"

# 5. GitHub 저장소 연결 (아래의 URL은 당신의 저장소 URL로 변경)
git remote add origin https://github.com/[당신의username]/qube-mvp.git

# 6. 메인 브랜치로 푸시
git branch -M main
git push -u origin main
```

### 예시 (실제 값으로 변경)
```powershell
cd D:\Users\장우진\dev26
git init
git config user.name "JangWoojin"
git config user.email "wj@example.com"
git add .
git commit -m "Initial MVP commit"
git remote add origin https://github.com/wj123/qube-mvp.git
git branch -M main
git push -u origin main
```

---

## 3단계: Streamlit Cloud 배포

### A. Streamlit Cloud 접속
1. https://streamlit.io/cloud 이동
2. **"Sign in"** 클릭
3. **GitHub으로 로그인**
4. Streamlit이 GitHub 접근 권한 요청 → **"Authorize"** 승인

### B. 앱 배포
1. **"New app"** 클릭
2. 설정:
   - **Repository**: `[username]/qube-mvp` 선택
   - **Branch**: `main`
   - **Main file path**: `mvp.py`
3. **"Deploy!"** 클릭

### C. Secrets 설정 (API 키)
1. 배포 후 앱 화면 우상단 **⋮ (메뉴)** → **Settings** 클릭
2. 좌측 **"Secrets"** 클릭
3. 아래 텍스트를 입력:
```toml
gemini_api_key = "AIzaSyARIHAyfjsPit6--Oe20V9EX_mASrZK5FM"
```
4. **"Save"** 클릭

### 완료! 🎉
몇 초 후 앱이 재시작되고, 상단의 **"Share"** 버튼에서 공개 URL 확인 가능:
```
https://[random-name]-mvp.streamlit.app
```

이 URL을 모바일에서 열면 앱 접속 가능합니다!

---

## ⚠️ 주의사항

### GitHub Push 할 때 제외될 파일들 (.gitignore):
- `secrets.toml` ❌ 클라우드 배포용 (Streamlit Secrets에서 관리)
- `.env` ❌ 로컬 개발용
- `__pycache__/` ❌ 자동 생성
- CSV 파일 ❌ 너무 큼

### API 키 노출 주의! 🔒
- GitHub에는 절대 API 키를 커밋하지 마세요
- Streamlit Cloud의 **Secrets** 탭에서만 관리하세요
- `.gitignore`에 `secrets.toml`과 `.env` 포함됨

### 배포 후 에러 발생 시:
1. **"Rerun"** 버튼 클릭
2. 여전히 오류 → 앱 우상단 **⋮** → **View logs** 확인
3. `.env` 또는 `secrets.toml` 관련 → API 키 재확인

---

## 🔗 유용한 링크

- [Streamlit Cloud 문서](https://docs.streamlit.io/streamlit-cloud/deploy-your-app)
- [GitHub 저장소 생성 가이드](https://docs.github.com/en/get-started/quickstart/create-a-repo)
- [Google Generative AI API 키 발급](https://makersuite.google.com/app/apikey)

---

## 완료 확인 체크리스트

- [ ] GitHub 저장소 생성
- [ ] 로컬에서 `git push` 완료
- [ ] Streamlit Cloud에서 배포 시작
- [ ] API 키를 Secrets에 입력
- [ ] 공개 URL에서 앱 열림 확인
- [ ] 모바일에서 접속 테스트

완료되면 모바일에서 **어디서든** 앱 접속 가능합니다! 🌍📱
