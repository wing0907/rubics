#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit Cloud 자동 배포 스크립트
GitHub 저장소를 생성하고 코드를 푸시합니다
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description=""):
    """명령어 실행"""
    if description:
        print(f"\n✓ {description}")
    print(f"  실행: {cmd}\n")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr and result.returncode != 0:
            print(f"❌ 에러: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"❌ 명령어 실행 실패: {e}")
        return False

def main():
    print("=" * 70)
    print("🚀 Streamlit Cloud 자동 배포 스크립트")
    print("=" * 70)
    
    # GitHub 정보 입력
    print("\n📋 정보 입력:")
    github_username = input("GitHub 사용자명: ").strip()
    github_email = input("GitHub 이메일: ").strip()
    repo_name = input("저장소 이름 (기본값: qube-mvp): ").strip() or "qube-mvp"
    github_url = f"https://github.com/{github_username}/{repo_name}.git"
    
    # 현재 디렉토리
    repo_dir = Path.cwd()
    
    print(f"\n📂 작업 디렉토리: {repo_dir}")
    print(f"🔗 GitHub URL: {github_url}")
    
    # 1. Git 초기화
    if not Path(".git").exists():
        if not run_command("git init", "1/5: Git 저장소 초기화"):
            return False
    
    # 2. Git 설정
    run_command(f'git config user.name "{github_username}"', "2/5: Git 사용자명 설정")
    run_command(f'git config user.email "{github_email}"', "       Git 이메일 설정")
    
    # 3. 파일 추가
    if not run_command("git add .", "3/5: 파일 추가 (add)"):
        return False
    
    # 4. 커밋
    if not run_command('git commit -m "Initial MVP commit"', "4/5: 커밋 생성"):
        return False
    
    # 5. GitHub 저장소 연결 및 푸시
    if not run_command(f'git remote add origin {github_url}', "5/5: GitHub 저장소 연결"):
        # 이미 연결된 경우
        run_command(f'git remote set-url origin {github_url}', "       (기존 원격 저장소 변경)")
    
    if not run_command("git branch -M main", "       메인 브랜치 이름 설정"):
        return False
    
    if not run_command("git push -u origin main", "       GitHub에 푸시"):
        return False
    
    # 완료
    print("\n" + "=" * 70)
    print("✅ 완료!")
    print("=" * 70)
    print(f"""
📍 다음 단계:

1. GitHub 저장소 확인
   → {github_url}

2. Streamlit Cloud 배포
   → https://streamlit.io/cloud 접속
   → "New app" 클릭
   → Repository: {github_username}/{repo_name}
   → Branch: main
   → Main file path: mvp.py
   → "Deploy!" 클릭

3. API 키 설정
   → 배포 완료 후 앱의 ⋮ → Settings → Secrets
   → 아래 입력:
   
   gemini_api_key = "your-api-key"

4. 완료!
   → 모바일에서 공개 URL로 접속 가능 🎉

""")

if __name__ == "__main__":
    main()
