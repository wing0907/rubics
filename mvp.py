# =========================================================
# Rubics MVP - 모바일 학습 AI 어시스턴트
# =========================================================
# 심플한 이미지 질문 AI 답변 서비스
#
# 기능:
# 1. 이미지 업로드 → 즉시 AI 분석
# 2. 유사 문제 추천
# 3. 학습 내용 요약
#
# UI: Claude 스타일 채팅 인터페이스
# =========================================================

import sys
import subprocess
from pathlib import Path
import os
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from PIL import Image
import io

# ===== 보안: API Key 관리 =====
import streamlit as st

# API Key 우선순위: 1. Streamlit Secrets (클라우드) > 2. 환경변수 (로컬)
def get_secure_api_key():
    """보안 강화: API Key 로드 (Streamlit Secrets 우선)"""
    try:
        if "gemini_api_key" in st.secrets:
            return st.secrets["gemini_api_key"]
    except:
        pass
    
    # 로컬 개발용 (프로덕션에서는 사용 안 함)
    if os.getenv("GEMINI_API_KEY_wj"):
        return os.getenv("GEMINI_API_KEY_wj")
    
    return None

GEMINI_API_KEY = get_secure_api_key()

# ===== 패키지 설치 =====
packages_needed = []
try:
    import google.generativeai as genai
except:
    packages_needed.append("google-generativeai")
try:
    import cv2
except:
    packages_needed.append("opencv-python")
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except:
    packages_needed.append("scikit-learn")

if packages_needed:
    cmd = [sys.executable, "-m", "pip", "install", "-q"] + packages_needed
    try:
        subprocess.check_call(cmd)
    except:
        pass

import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===== 설정 =====
GEMINI_MODEL = "gemini-2.0-flash"
OUTPUT_DIR = Path(r"D:\Users\장우진\dev26\qube_out_mvp")
OUTPUT_DIR.mkdir(exist_ok=True)

# Gemini 초기화
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception as e:
        st.error(f"❌ Gemini 초기화 오류: {e}")

# ===== 데이터 로더 =====
@st.cache_resource
def load_master_answers():
    """마스터 답변 데이터 로드 (캐싱)"""
    try:
        data_dir = Path(r"D:\Users\Qube\0. 데이터\1. 분석\데이터분석_최종\decoded_messages")
        if not data_dir.exists():
            return []
        
        all_data = []
        for csv_file in sorted(data_dir.glob("*.csv")):
            df = pd.read_csv(csv_file, encoding='utf-8-sig')
            all_data.append(df)
        
        if not all_data:
            return []
        
        df_combined = pd.concat(all_data, ignore_index=True)
        
        # 마스터 답변만 추출
        master_data = df_combined[df_combined['speaker_role'] == 'master'].copy()
        master_data = master_data[['QM_QST_NO', 'content', 'DomName']].dropna()
        
        problems = []
        for idx, row in master_data.iterrows():
            problems.append({
                'id': str(row['QM_QST_NO']),
                'answer': str(row['content']),
                'domain': str(row['DomName'])
            })
        
        return problems
    except Exception as e:
        st.warning(f"⚠️ 데이터 로드 실패: {e}")
        return []

# ===== 유사 문제 검색 =====
@st.cache_resource
def build_problem_index(problems):
    """TF-IDF 인덱스 구축"""
    if not problems:
        return None, None
    
    texts = [p['answer'] for p in problems]
    vectorizer = TfidfVectorizer(max_features=200, min_df=1)
    
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
        return vectorizer, tfidf_matrix
    except:
        return None, None

def search_similar_problems(query, problems, vectorizer, tfidf_matrix, top_k=3):
    """유사 문제 검색"""
    if not vectorizer or tfidf_matrix is None:
        return []
    
    try:
        query_vec = vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, tfidf_matrix)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:
                results.append({
                    'answer': problems[idx]['answer'][:200],
                    'domain': problems[idx]['domain'],
                    'score': float(similarities[idx])
                })
        
        return results
    except:
        return []

# ===== Gemini API 호출 =====
def analyze_image_with_gemini(image: Image.Image, question: str = ""):
    """이미지 분석 및 답변 생성"""
    if not GEMINI_API_KEY:
        return None, "❌ API Key 설정 필요합니다. Streamlit Cloud의 Secrets에서 설정하세요."
    
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        prompt = f"""사용자가 업로드한 이미지를 분석하고 학습을 돕는 답변을 제공하세요.

{f'추가 질문: {question}' if question else ''}

다음 포맷으로 답변하세요:
1. 📌 이미지에서 인식된 내용
2. 📚 핵심 개념 설명
3. 💡 학습 팁"""
        
        response = model.generate_content([image, prompt])
        return response.text, None
    except Exception as e:
        error_msg = str(e)
        if "Quota exceeded" in error_msg:
            return None, "⚠️ API 할당량 초과. 내일 다시 시도하세요."
        elif "401" in error_msg or "API key" in error_msg:
            return None, "❌ API Key 오류. Streamlit Cloud의 Secrets 설정을 확인하세요."
        return None, f"❌ 오류: {error_msg[:100]}"

# ===== Streamlit UI =====
st.set_page_config(page_title="Rubics", layout="wide", initial_sidebar_state="collapsed")

# CSS 스타일 (모바일 최적화)
st.markdown("""
<style>
    /* 여백 최소화 */
    .main { padding: 0.5rem; }
    .stContainer { max-width: 100%; }
    
    /* 채팅 스타일 */
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        font-size: 0.95rem;
    }
    .user-message {
        background-color: #e3f2fd;
        text-align: right;
    }
    .ai-message {
        background-color: #f5f5f5;
        text-align: left;
    }
    
    /* 모바일 최적화 */
    @media (max-width: 768px) {
        .main { padding: 0.25rem; }
        .stMarkdown { font-size: 0.9rem; }
        .stButton > button { width: 100%; padding: 0.5rem; }
        .stTextInput > div > div > input { font-size: 1rem; }
        .stFileUploader { padding: 0.5rem; }
    }
</style>
""", unsafe_allow_html=True)

# ===== 헤더 =====
st.title("📚 Rubics")
st.markdown("**이미지로 배우는 AI 학습 도우미**")

# ===== 세션 상태 =====
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "current_image" not in st.session_state:
    st.session_state.current_image = None

# ===== 이미지 업로드 =====
st.subheader("📸 이미지 업로드")

uploaded_file = st.file_uploader(
    "문제 사진을 업로드하세요",
    type=["jpg", "jpeg", "png", "gif", "webp"],
    label_visibility="collapsed"
)

if uploaded_file:
    # 이미지 저장 및 표시
    image = Image.open(uploaded_file)
    st.session_state.current_image = image
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.image(image, use_column_width=True, caption="업로드된 이미지")

# ===== 채팅 영역 =====
st.subheader("💬 질문 및 답변")

# 채팅 히스토리 표시
chat_container = st.container()
with chat_container:
    for i, msg in enumerate(st.session_state.chat_history):
        if msg["role"] == "user":
            st.markdown(f"""
            <div class='chat-message user-message'>
                <strong>👤 You:</strong><br>{msg['content']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='chat-message ai-message'>
                <strong>🤖 Rubics:</strong><br>{msg['content']}
            </div>
            """, unsafe_allow_html=True)

# ===== 입력 영역 =====
st.divider()

# 텍스트 입력 (질문)
user_question = st.text_input(
    "질문을 입력하고 Enter를 누르세요",
    placeholder="예: 이 문제는 어떻게 풀어?",
    label_visibility="collapsed"
)

# Enter 누르면 자동 분석
if user_question or st.session_state.current_image:
    if st.session_state.current_image and user_question:
        # 사용자 메시지 추가
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_question
        })
        
        # AI 분석
        with st.spinner("🔍 분석 중..."):
            answer, error = analyze_image_with_gemini(
                st.session_state.current_image,
                user_question
            )
        
        if error:
            st.error(error)
        else:
            # AI 응답 추가
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer
            })
            
            # 유사 문제 검색
            problems = load_master_answers()
            if problems:
                vectorizer, tfidf_matrix = build_problem_index(problems)
                similar = search_similar_problems(user_question, problems, vectorizer, tfidf_matrix)
                
                if similar:
                    st.info("📚 **유사 문제**")
                    for i, prob in enumerate(similar, 1):
                        st.write(f"{i}. [{prob['domain']}] {prob['answer']}")
        
        # 페이지 새로고침
        st.rerun()

# ===== 사이드바: 정보 =====
with st.sidebar:
    st.markdown("### ℹ️ 정보")
    st.markdown("""
    **Rubics**는 AI 기반 학습 도우미입니다.
    
    - 📸 이미지로 문제 분석
    - 🤖 AI가 설명해줍니다
    - 📚 유사 문제 추천
    
    **사용법:**
    1. 문제 사진 업로드
    2. 질문 입력
    3. Enter 누르기
    4. 답변 받기!
    """)
    
    st.divider()
    
    if st.button("🗑️ 채팅 초기화", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.current_image = None
        st.rerun()

# ===== 하단 정보 =====
st.divider()
st.caption("🔒 API Key는 안전하게 보관됩니다. | Powered by Google Gemini")
