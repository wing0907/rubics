# 📱 모바일에서 Qube MVP 접근하기

## 🚀 빠른 시작

### PC (서버 실행)
```bash
$env:GEMINI_API_KEY_wj = "YOUR_API_KEY"
streamlit run mvp.py
```

### 모바일 (같은 WiFi)
1. PC 네트워크 IP 확인: **`10.1.0.59`**
2. 모바일 브라우저에서 접속:
   ```
   http://10.1.0.59:8501
   ```

---

## 📋 요구사항

- 🖥️ **PC와 모바일이 같은 WiFi 네트워크에 연결**
- 📱 **모바일 브라우저** (Chrome, Safari, Edge 등)
- 🔑 **Gemini API Key** (설정 탭에서 입력)

---

## ✨ 모바일 최적화 기능

✅ 반응형 레이아웃
✅ 터치 친화적 버튼
✅ 최소화된 사이드바
✅ 모바일 파일 업로드 지원
✅ 최적화된 텍스트 입력 필드

---

## 🔐 보안 참고사항

- **로컬 네트워크만 사용** - 인터넷 공개 X
- **API Key는 절대 공개하지 마세요**
- **민감한 학습 데이터 보호**

---

## 📱 모바일 접근 URL 형식

```
http://[PC_IP]:8501
```

### 예시:
- `http://10.1.0.59:8501` ← 현재 PC
- `http://192.168.1.100:8501` ← 다른 WiFi
- `http://172.16.0.50:8501` ← 회사 네트워크

---

## 🛠️ 문제 해결

### ❌ "연결할 수 없음"
- PC의 IP 주소 확인: `ipconfig` 실행
- 모바일이 같은 WiFi에 연결되어 있는지 확인
- 방화벽 확인 (포트 8501)

### ❌ 파일 업로드 실패
- 모바일 브라우저 캐시 삭제
- 다시 새로고침 (Ctrl+Shift+R 또는 강제 새로고침)
- 작은 파일(1MB 이하)부터 테스트

### ❌ API 오류
- ⚙️ 설정 탭에서 API Key 확인
- "API Key 확인" 버튼으로 유효성 검사
- Gemini API 할당량 확인

---

## 🌐 향후 개선 (선택사항)

### Streamlit Cloud 배포
- GitHub에 코드 업로드
- Streamlit Cloud에 연동
- 어디서나 접근 가능

### Docker 배포
```bash
docker build -t qube-mvp .
docker run -p 8501:8501 qube-mvp
```

### 클라우드 호스팅
- Heroku, Railway, Render 등

---

**Happy Learning! 📚✨**
