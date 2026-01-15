#!/usr/bin/env python3
"""
ngrok으로 Streamlit을 공개 인터넷에 노출시키는 스크립트
모바일에서 인터넷을 통해 접속 가능 (WiFi/네트워크 상관없음)
"""
import subprocess
import time
from pyngrok import ngrok

def main():
    print("=" * 60)
    print("🌐 ngrok Streamlit 터널 시작")
    print("=" * 60)
    
    # Streamlit이 이미 8501에서 실행중이면 그대로 사용
    # 아니면 새로 시작해야 함
    print("\n⚠️  주의: Streamlit이 이미 다른 터미널에서 실행중이어야 합니다!")
    print("   명령어: streamlit run mvp.py")
    print("\n계속하려면 엔터를 누르세요...")
    input()
    
    try:
        print("\n🔗 ngrok 터널 생성 중...")
        ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN")  # ngrok 계정이 없으면 설정 불필요
        
        # 포트 8501에 대한 터널 생성
        public_url = ngrok.connect(8501, "http")
        
        print("\n" + "=" * 60)
        print("✅ ngrok 터널 생성 완료!")
        print("=" * 60)
        print(f"\n📱 모바일에서 접속할 URL:\n   {public_url}\n")
        print("이 URL은 전 세계 어디서나 접속 가능합니다!")
        print("\n터미널을 닫지 마세요. Ctrl+C로 중단할 수 있습니다.")
        print("=" * 60)
        
        # 터널 유지
        ngrok_process = ngrok.get_ngrok_process()
        ngrok_process.proc.wait()
        
    except Exception as e:
        print(f"❌ 에러: {e}")
        print("\n해결 방법:")
        print("1. ngrok 가입: https://ngrok.com")
        print("2. 토큰 얻기: https://dashboard.ngrok.com/auth")
        print("3. 위의 YOUR_NGROK_AUTH_TOKEN 부분에 붙여넣기")

if __name__ == "__main__":
    main()
