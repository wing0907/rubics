#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Gemini 사용 가능한 모델 확인"""

import os
from pathlib import Path

# .env 파일 로드
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

import google.generativeai as genai

api_key = os.environ.get("GEMINI_API_KEY_wj")
if not api_key:
    print("❌ GEMINI_API_KEY_wj 환경변수가 설정되지 않았습니다")
    exit(1)

genai.configure(api_key=api_key)

print("=" * 60)
print("Gemini 사용 가능한 모델 목록")
print("=" * 60)

try:
    models = genai.list_models()
    available_models = []
    
    for m in models:
        model_name = m.name
        methods = list(m.supported_generation_methods) if hasattr(m, 'supported_generation_methods') else []
        
        if 'generateContent' in methods:
            available_models.append(model_name)
            print(f"✓ {model_name}")
    
    print("=" * 60)
    print(f"\n✅ 총 {len(available_models)}개 모델 사용 가능\n")
    
    # 추천 모델
    print("📌 권장 모델:")
    recommended = [
        "models/gemini-1.5-flash",
        "models/gemini-2.0-flash",
        "models/gemini-pro",
        "models/gemini-1.5-pro"
    ]
    
    for model in recommended:
        if any(model in m for m in available_models):
            print(f"  ✓ {model}")
            
except Exception as e:
    print(f"❌ 오류: {e}")
    import traceback
    traceback.print_exc()
