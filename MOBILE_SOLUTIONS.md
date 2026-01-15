# 📱 ngrok으로 모바일 접근하기 (WiFi 없이도 가능!)

## 🌐 ngrok 설치 및 사용

### 1️⃣ ngrok 설치
```bash
pip install pyngrok
```

### 2️⃣ Streamlit + ngrok 스크립트

`run_with_ngrok.py` 파일 생성:

```python
import streamlit as st
from pyngrok import ngrok

# Streamlit 실행
st.set_page_config(page_title="Qube", page_icon="📚")
st.title("Qube MVP with ngrok")

# ngrok 터널 생성 (포트 8501)
public_url = ngrok.connect(8501)
print(f"✅ 공개 URL: {public_url}")

# 나머지 Streamlit 코드...
```

### 3️⃣ 실행
```bash
$env:GEMINI_API_KEY_wj = "YOUR_KEY"
streamlit run run_with_ngrok.py
```

### 4️⃣ 터미널에 출력되는 공개 URL로 모바일에서 접근!

---

## 장점
✅ WiFi 없이도 전 세계 어디서나 접근 가능
✅ 설정 불필요 (자동으로 공개 URL 생성)
✅ https 자동 암호화

## 단점
✗ 인터넷 속도에 영향을 받음
✗ 무료 버전은 8시간 제한

---

## 간단한 테스트 (현재 상황)

1. **모바일 브라우저에서:**
   ```
   http://10.1.0.59:8501
   ```

2. **여전히 안 되면:**
   ```bash
   # PC 방화벽 비활성화 (테스트용)
   netsh advfirewall set allprofiles state off
   
   # 다시 시도한 후:
   netsh advfirewall set allprofiles state on
   ```

3. **ngrok 사용:**
   ```bash
   pip install pyngrok
   ```

---

**어느 방법이 원하시나요?** 🚀
