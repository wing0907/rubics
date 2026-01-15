#!/bin/bash
# 모바일 접근 문제 해결 스크립트

echo "=== Streamlit 모바일 접근 문제 해결 ==="
echo ""

# 1. PC IP 확인
echo "1️⃣  PC 네트워크 설정 확인"
ipconfig | findstr "IPv4"
echo ""

# 2. 방화벽 상태 확인
echo "2️⃣  방화벽 상태 확인"
netsh advfirewall show allprofiles | findstr "상태"
echo ""

# 3. 포트 8501 확인
echo "3️⃣  포트 8501 리스닝 확인"
netstat -ano | findstr "8501"
echo ""

# 4. Streamlit 프로세스 확인
echo "4️⃣  Streamlit 프로세스 확인"
Get-Process | findstr "streamlit" | Select-Object Name, ID
echo ""

echo "=== 문제 해결 팁 ==="
echo "✓ PC와 모바일이 같은 WiFi에 연결되어 있는가?"
echo "✓ 방화벽이 포트 8501을 차단하지 않는가?"
echo "✓ PC의 실제 IP 주소를 사용하는가? (0.0.0.0 아님)"
echo "✓ 모바일에서 http:// 프로토콜을 사용하는가? (https 아님)"
