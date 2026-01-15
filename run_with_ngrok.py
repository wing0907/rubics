"""
Streamlit MVP with ngrok 지원
모바일에서 어디서나 접근 가능
"""

import os
import sys
import streamlit as st

# ngrok 터널 생성 (선택사항)
try:
    from pyngrok import ngrok
    
    # ngrok이 이미 실행 중이 아니면 시작
    if not os.environ.get('NGROK_ACTIVE'):
        public_url = ngrok.connect(8501)
        os.environ['NGROK_ACTIVE'] = 'true'
        
        # 공개 URL을 파일에 저장
        with open('NGROK_URL.txt', 'w') as f:
            f.write(f"모바일 접근 URL: {public_url}\n")
        
        st.info(f"🌐 **공개 URL:** {public_url}")
except Exception as e:
    st.warning(f"ngrok 사용 불가: {e}")

# 여기서부터는 mvp.py의 나머지 코드를 실행
import subprocess
result = subprocess.run([sys.executable, 'mvp.py'], cwd=os.path.dirname(__file__))
sys.exit(result.returncode)
