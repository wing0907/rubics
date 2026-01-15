# 📱 Rubics - 모바일 학습 플랫폼

AI 기반 이미지 문제 풀이 및 맞춤형 학습 도우미 애플리케이션입니다.

## 🎯 핵심 기능

1. **🖼️ 이미지 질문 답변** - 카메라로 찍은 문제를 Gemini AI로 분석
2. **🔍 유사 문제 검색** - 마스터 답변을 바탕으로 유사 문제 추천
3. **📝 답변 형태 선택** - 간단한/자세한 설명 중 선택
4. **📚 맞춤형 학습 콘텐츠** - 세션 기반 학습 요약
5. **🔄 복습 스케줄** - Ebbinghaus 간격 반복법 기반 복습 추적

## 🚀 빠른 시작

### 로컬 실행

```bash
# 1. 저장소 클론
git clone <repository-url>
cd mvp

# 2. 패키지 설치
pip install -r requirements.txt

# 3. API 키 설정 (로컬)
export GEMINI_API_KEY_wj="your-api-key"  # 또는 .env 파일에 작성

# 4. Streamlit 실행
streamlit run mvp.py
```

### Streamlit Cloud 배포

1. https://streamlit.io/cloud 방문
2. GitHub 저장소 선택
3. **App settings** → **Secrets** → 아래 입력:
   ```toml
   gemini_api_key = "your-api-key"
   ```
4. **Deploy** 클릭

## 🔐 API 키 설정

### 로컬 개발
`.env` 파일 생성:
```
GEMINI_API_KEY_wj=your-api-key
```

### 클라우드 배포 (Streamlit Cloud)
1. Streamlit Cloud의 App settings 이동
2. **Secrets** 탭에서 추가:
```toml
gemini_api_key = "your-api-key"
```

## 📋 요구사항

- Python 3.8+
- Streamlit 1.51.0
- Google Generative AI API 키 (무료)
- OpenCV, scikit-learn, pandas

## 🛠️ 기술 스택

| 구분 | 기술 |
|------|------|
| **프론트엔드** | Streamlit |
| **AI 모델** | Google Gemini 2.0 Flash |
| **이미지 처리** | OpenCV, PIL |
| **텍스트 검색** | scikit-learn TF-IDF |
| **배포** | Streamlit Cloud |

## 📱 모바일 접근

### WiFi 핫스팟
- PC에서 WiFi 핫스팟 활성화
- 모바일에서 `http://192.168.137.1:8501` 접속

### 클라우드 배포
- Streamlit Cloud 배포 후 공개 URL로 접속
- 인터넷만 있으면 어디서든 가능

## 🔧 데이터 소스

- **master_answers**: 교육 전문가가 작성한 답변 (decoded_messages CSV)
- **student_questions**: 학생 질문 데이터
- **DomName**: 도메인 분류 (수학, 영어, 과학 등)

## ⚠️ 주의사항

- Gemini API는 무료 티어의 일일 할당량이 제한됨
- 이미지 처리는 데이터 로컬 저장소에 의존함
- 클라우드 배포 시 서버리스 환경이므로 큰 파일 처리 주의

## 📞 문제 해결

### API 키 오류
```
→ Streamlit Cloud의 Secrets에서 API 키 확인
→ 또는 로컬에서 .env 파일에서 환경변수 확인
```

### 이미지 업로드 실패
```
→ .streamlit/config.toml에서 maxUploadSize 확인 (현재 500MB)
→ 500MB 이상인 파일은 압축 후 시도
```

### Gemini API 할당량 초과
```
→ 다음 날 동일 시간에 다시 시도
→ Google Cloud Console에서 API 활성화 확인
→ 필요시 유료 계정 업그레이드
```

## 📄 라이선스

MIT License

## 👥 개발자

- Qube Learning Platform Team

---

**더 많은 도움**: [Streamlit 문서](https://docs.streamlit.io) | [Google Gemini API](https://ai.google.dev)
