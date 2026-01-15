#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ngrok으로 Streamlit 공개 URL 생성
모바일이 어떤 네트워크에 있어도 이 URL로 접속 가능
"""
import subprocess
import sys
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
from pyngrok import ngrok

print("=" * 70)
print("🌐 ngrok 터널 시작 - Streamlit을 인터넷에 공개")
print("=" * 70)

try:
    print("\n⚙️  ngrok 설정 중...")
    # 인증 토큰 설정 (선택사항 - 없어도 작동하지만 제한있음)
    # ngrok.set_auth_token("YOUR_TOKEN_HERE")
    
    print("🔗 localhost:8501 → 공개 URL로 연결 중...\n")
    
    # TCP가 아닌 HTTP 프로토콜로 연결
    public_url = ngrok.connect(8501, "http")
    
    print("=" * 70)
    print("✅ 성공! 모바일에서 이 주소로 접속하세요:")
    print("=" * 70)
    print(f"\n   📱 {public_url}\n")
    print("=" * 70)
    print("\n이 URL은:")
    print("  ✓ 전 세계 어디서나 접속 가능")
    print("  ✓ WiFi와 유선 네트워크 상관없음")
    print("  ✓ 이 터미널을 열어있는 동안 유효")
    print("\n계속 진행 중... (Ctrl+C로 종료)\n")
    print("=" * 70)
    
    # 터널 유지
    ngrok.get_ngrok_process().proc.wait()
    
except KeyboardInterrupt:
    print("\n\n⏹️  ngrok 종료됨")
    sys.exit(0)
except Exception as e:
    print(f"\n❌ 에러 발생: {e}")
    print("\n해결 방법:")
    print("  1. ngrok이 설치되었는지 확인: pip install pyngrok")
    print("  2. 인터넷 연결 확인")
    sys.exit(1)
