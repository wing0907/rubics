#!/usr/bin/env python3
"""
모바일 WiFi에서 접속 가능한 임시 솔루션
ngrok 없이 간단한 HTTP 서버 + 모바일 친화적 설정
"""

import os
import sys

# Streamlit 설정 파일 생성
os.makedirs(".streamlit", exist_ok=True)

config_content = """[client]
showErrorDetails = true
toolbarMode = "minimal"

[server]
headless = true
port = 8501
address = 0.0.0.0
enableCORS = false
enableXsrfProtection = false
maxUploadSize = 500

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#F5F5F5"
secondaryBackgroundColor = "#E0E0E0"
textColor = "#262730"
font = "sans serif"

[logger]
level = "warning"
"""

config_path = ".streamlit/config.toml"
with open(config_path, "w", encoding="utf-8") as f:
    f.write(config_content)

print("✅ Streamlit 설정 완료")
print("\n🚀 모바일 실행 가이드:")
print("=" * 70)
print("\n방법 1: 같은 WiFi 네트워크 연결")
print("  1. PC의 WiFi를 켜고 모바일과 같은 WiFi에 연결")
print("  2. PC가 WiFi에서 받은 IP 주소 확인: ipconfig")
print("  3. 모바일에서 http://[PC_IP]:8501 로 접속")
print("\n방법 2: Streamlit Cloud (추천)")
print("  1. https://streamlit.io/cloud 에서 무료 계정 생성")
print("  2. GitHub에 코드 업로드")
print("  3. Streamlit Cloud에서 배포")
print("  4. 모바일에서 공개 URL로 접속 (전 세계)")
print("\n방법 3: 로컬 실행 (현재)")
print("  1. 같은 PC에서만 접속 가능: http://localhost:8501")
print("\n" + "=" * 70)
print("\n지금 시작: streamlit run mvp.py")
