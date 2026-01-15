#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""모든 import 검증"""

import sys

print("=" * 50)
print("필수 패키지 import 검증")
print("=" * 50)

packages = {
    'cv2': 'OpenCV',
    'pandas': 'Pandas',
    'numpy': 'NumPy',
    'PIL': 'Pillow',
    'sklearn': 'scikit-learn',
    'google.generativeai': 'Google Generative AI',
    'streamlit': 'Streamlit'
}

failed = []
for pkg, name in packages.items():
    try:
        __import__(pkg)
        print(f"✓ {name:30} - 설치됨")
    except ImportError as e:
        print(f"✗ {name:30} - 미설치")
        failed.append(pkg)

print("=" * 50)

if failed:
    print(f"\n❌ {len(failed)}개 패키지 미설치:")
    for pkg in failed:
        print(f"  - pip install {pkg}")
    sys.exit(1)
else:
    print("\n✅ 모든 패키지 설치 완료!")
    print("\n이제 Streamlit으로 실행할 수 있습니다:")
    print("  streamlit run mvp.py")
    sys.exit(0)
