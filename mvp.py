# =========================================================
# Qube 모바일 앱 MVP - 문제 기반 학습 플랫폼
# =========================================================
# 핵심 기능:
# 1. 이미지 질문 → Gemini 답변
# 2. 유사 문제 검색 (ORB/SIFT + 특성 기반 매칭)
# 3. 답변 형태 선택 (간단/자세함)
# 4. 맞춤형 학습 콘텐츠 요약
# 5. 복습 스케줄 알림 (Spaced Repetition)
#
# 모델: Gemini (이미지 분석)
# 일반: YOLOv11 (물체 인식)
# 프리미엄: SAM3 (매직 지우개, 오답노트)
# =========================================================

import sys
import subprocess
from pathlib import Path
import os
import json
import pickle
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from PIL import Image
import io
import base64
import hashlib

# .env 파일 로드
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    # dotenv가 없으면 자동으로 .env 파일 읽기
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

# OpenCV 안전 import
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("⚠️ OpenCV가 설치되지 않았습니다. 이미지 특성 검색이 제한됩니다.")

# ===== 패키지 자동 설치 =====
def pip_install(pkgs):
    if not pkgs:
        return
    cmd = [sys.executable, "-m", "pip", "install", "-q"] + pkgs
    try:
        subprocess.check_call(cmd)
    except Exception as e:
        print(f"⚠️ 패키지 설치 실패: {e}")

packages_needed = []
try:
    import streamlit
except:
    packages_needed.append("streamlit")
try:
    import google.generativeai
except:
    packages_needed.append("google-generativeai")
try:
    import sklearn
except:
    packages_needed.append("scikit-learn")
if not HAS_CV2:
    packages_needed.append("opencv-python")
try:
    from PIL import Image
except:
    packages_needed.append("pillow")

if packages_needed:
    print(f"📦 설치 중: {packages_needed}")
    pip_install(packages_needed)

import streamlit as st
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ===== 기본 설정 =====
class Config:
    # 데이터 경로
    DATA_ROOT = Path(r"D:\Users\Qube\0. 데이터")
    ANALYSIS_DIR = DATA_ROOT / "1. 분석"
    SAMPLING_DIR = DATA_ROOT / "2. 샘플링"
    TRAINING_POOL_DIR = DATA_ROOT / "3. 학습 후보 풀"
    
    # 출력 디렉토리
    OUTPUT_DIR = Path(r"D:\Users\장우진\dev26\qube_out_mvp")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Gemini API (Streamlit secrets > 환경변수 순서로 읽기)
    @staticmethod
    def get_api_key():
        # 1순위: Streamlit secrets (클라우드 배포용)
        try:
            if "gemini_api_key" in st.secrets:
                return st.secrets["gemini_api_key"]
        except:
            pass
        
        # 2순위: 환경변수 (로컬 개발용)
        api_key = os.environ.get("GEMINI_API_KEY_wj", "")
        if api_key:
            return api_key
        
        return ""
    
    GEMINI_API_KEY = get_api_key.__func__()
    
    # Gemini 모델 설정
    # 지원하는 모델: gemini-2.0-flash, gemini-2.5-flash, gemini-pro 등
    GEMINI_MODEL = "gemini-2.0-flash"
    
    # 기본값
    DEFAULT_PORT = 8501
    MAX_SIMILAR_PROBLEMS = 5


# Gemini 초기화
if Config.GEMINI_API_KEY:
    try:
        genai.configure(api_key=Config.GEMINI_API_KEY)
    except Exception as e:
        print(f"⚠️ Gemini 초기화 오류: {e}")
else:
    st.warning("⚠️ Gemini API 키가 설정되지 않았습니다. 설정에서 API 키를 입력하세요.")


# ===== 데이터 로더 =====
class DataManager:
    """분석, 샘플링, 학습 데이터 로드 및 관리"""
    
    def __init__(self):
        self.problems = []
        self.metadata = {}
        self._load_data()
    
    def _load_data(self):
        """CSV 및 이미지 데이터 로드 - 마스터 답변 포함"""
        # 1. decoded_messages에서 마스터 답변 로드
        try:
            decoded_dir = Config.ANALYSIS_DIR / "데이터분석_최종" / "decoded_messages"
            if decoded_dir.exists():
                decoded_files = sorted(decoded_dir.glob("decoded_messages_part*.csv"))
                if decoded_files:
                    dfs = []
                    for f in decoded_files:
                        df = pd.read_csv(f, encoding='utf-8-sig')
                        dfs.append(df)
                    all_decoded = pd.concat(dfs, ignore_index=True)
                    
                    # 문제별로 마스터 답변 추출
                    self.problems = []
                    for qid, group in all_decoded.groupby('QM_QST_NO'):
                        master_msgs = group[group['speaker_role'] == 'master']
                        student_msgs = group[group['speaker_role'] == 'student']
                        
                        problem = {
                            'id': str(qid),
                            'QM_QST_NO': qid,
                            'DomName': group['DomName'].iloc[0] if len(group) > 0 else '',
                            'SubName': group['SubName'].iloc[0] if len(group) > 0 else '',
                            'class': group['class_value'].iloc[0] if len(group) > 0 else '',
                            'master_answer': ' '.join(master_msgs['qst_text_decoded'].tolist()) if len(master_msgs) > 0 else '',
                            'student_question': ' '.join(student_msgs['qst_text_decoded'].tolist()) if len(student_msgs) > 0 else '',
                            'has_image': group['has_image'].iloc[0] if len(group) > 0 else False,
                        }
                        if problem['master_answer']:  # 마스터 답변이 있는 경우만
                            self.problems.append(problem)
                    
                    print(f"✓ 마스터 답변: {len(self.problems)}개 문제 로드됨 (decoded_messages에서)")
                else:
                    print(f"⚠️ decoded_messages 폴더가 비어있습니다: {decoded_dir}")
                    self._create_sample_data()
            else:
                print(f"⚠️ decoded_messages 폴더가 없습니다: {decoded_dir}")
                self._create_sample_data()
        except Exception as e:
            print(f"⚠️ 마스터 답변 로드 실패: {e}")
            self._create_sample_data()
        
        # 2. 분석 데이터 (메타데이터)
        try:
            analysis_files = list(Config.ANALYSIS_DIR.glob("데이터분석_최종/*.csv"))
            if analysis_files:
                self.analysis_data = pd.concat([
                    pd.read_csv(f, encoding='utf-8-sig') 
                    for f in analysis_files
                ], ignore_index=True)
                print(f"✓ 분석 메타데이터: 로드됨")
        except Exception as e:
            print(f"⚠️ 분석 데이터 로드 실패: {e}")
        
        # 3. 샘플링 데이터 (대표 예시)
        try:
            sampling_files = list(Config.SAMPLING_DIR.glob("*.csv"))
            if sampling_files:
                self.sampling_data = pd.concat([
                    pd.read_csv(f, encoding='utf-8-sig')
                    for f in sampling_files
                ], ignore_index=True)
                print(f"✓ 샘플링 데이터: 로드됨")
        except Exception as e:
            print(f"⚠️ 샘플링 데이터 로드 실패: {e}")
        
        # 4. 학습 후보 풀 (YOLOv11, SAM3 전용)
        try:
            pool_files = list(Config.TRAINING_POOL_DIR.glob("*.csv"))
            if pool_files:
                self.training_pool = pd.concat([
                    pd.read_csv(f, encoding='utf-8-sig')
                    for f in pool_files
                ], ignore_index=True)
                print(f"✓ 학습 후보 풀: 로드됨")
        except Exception as e:
            print(f"⚠️ 학습 후보 풀 로드 실패: {e}")
    
    def _create_sample_data(self):
        """더미 샘플 데이터 생성 (테스트용)"""
        self.problems = [
            {
                'id': f'problem_{i}',
                'content': f'Sample problem {i}. This is a test problem for demonstrating the system.',
                'category': f'Category {i % 3}',
                'difficulty': ['easy', 'medium', 'hard'][i % 3]
            }
            for i in range(5)  # 최소 5개의 샘플 데이터
        ]
        print(f"⚠️ 더미 데이터로 {len(self.problems)}개 샘플 생성됨")
    
    def get_all_problems(self) -> List[Dict]:
        """모든 문제 반환"""
        return self.problems
    
    def get_problem_by_id(self, problem_id: str) -> Optional[Dict]:
        """ID로 문제 조회"""
        for p in self.problems:
            if p.get('id') == problem_id:
                return p
        return None


# ===== 유사 문제 검색 (ORB/SIFT) =====
class SimilarProblemFinder:
    """이미지 특성 기반 + 텍스트 기반 유사 문제 검색"""
    
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        self.orb = cv2.ORB_create(nfeatures=500) if HAS_CV2 else None
        self.sift = cv2.SIFT_create() if HAS_CV2 else None
        # TfidfVectorizer를 더 안전하게 설정 (min_df=1, stop_words 제거)
        self.tfidf = TfidfVectorizer(
            max_features=200,
            min_df=1,
            max_df=0.95,
            ngram_range=(1, 1),
            lowercase=True
        )
        self._build_text_index()
    
    def _build_text_index(self):
        """문제 및 마스터 답변 기반 TF-IDF 색인 구축"""
        problems = self.data_manager.get_all_problems()
        
        # 마스터 답변 + 학생 질문 합치기
        texts = []
        for p in problems:
            # 마스터 답변이 우선순위
            text = str(p.get('master_answer', '')) or str(p.get('student_question', ''))
            if not text:
                text = str(p.get('content', ''))  # 폴백
            texts.append(text)
        
        # 공백이 아닌 텍스트만 필터링
        texts = [t.strip() for t in texts if t.strip()]
        
        if texts and len(texts) > 1:
            try:
                self.tfidf_matrix = self.tfidf.fit_transform(texts)
                print(f"✓ TF-IDF 색인: {len(texts)}개 문제 색인됨 (마스터 답변 포함)")
            except ValueError as e:
                # empty vocabulary 에러 처리
                print(f"⚠️ TF-IDF 색인 구축 실패: {e}")
                print(f"   로드된 텍스트: {len(texts)}개 (최소 2개 필요)")
                self.tfidf_matrix = None
        else:
            print(f"⚠️ 텍스트 데이터 부족: {len(texts)}개 (최소 2개 필요)")
            self.tfidf_matrix = None
    
    def find_similar_by_text(self, query: str, top_k: int = 5) -> List[Dict]:
        """텍스트 기반 유사 문제 검색 (마스터 답변 포함)"""
        if self.tfidf_matrix is None:
            return []
        
        try:
            query_vec = self.tfidf.transform([query])
            scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]
            top_indices = np.argsort(scores)[-top_k:][::-1]
            
            problems = self.data_manager.get_all_problems()
            results = []
            
            for idx in top_indices:
                if scores[idx] > 0.05:  # 유사도 임계값 조정
                    problem = problems[idx]
                    results.append({
                        'id': problem.get('id', ''),
                        'QM_QST_NO': problem.get('QM_QST_NO', ''),
                        'DomName': problem.get('DomName', ''),
                        'SubName': problem.get('SubName', ''),
                        'class': problem.get('class', ''),
                        'similarity': float(scores[idx]),
                        'master_answer': problem.get('master_answer', ''),
                        'student_question': problem.get('student_question', ''),
                    })
            
            return results
        except Exception as e:
            print(f"⚠️ 유사 문제 검색 실패: {e}")
            return []
    
    def find_similar_by_image(self, image: np.ndarray, top_k: int = 5) -> List[Dict]:
        """이미지 기반 유사 문제 검색 (ORB/SIFT)"""
        if not HAS_CV2:
            return {'status': 'cv2_not_available', 'message': 'OpenCV가 설치되지 않았습니다'}
        
        try:
            # 업로드 이미지에서 특성 추출
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            
            # ORB 특성 (빠름)
            kp_orb, des_orb = self.orb.detectAndCompute(gray, None)
            # SIFT 특성 (정확도)
            kp_sift, des_sift = self.sift.detectAndCompute(gray, None)
            
            # 특성이 있는 경우에만 매칭 수행
            if des_orb is not None and len(kp_orb) > 3:
                return {
                    'orb_keypoints': len(kp_orb),
                    'sift_keypoints': len(kp_sift) if des_sift is not None else 0,
                    'status': 'success'
                }
            else:
                return {'status': 'no_features_found'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def find_similar_combined(self, image: np.ndarray, query_text: str = "", top_k: int = 5) -> List[Dict]:
        """이미지 + 텍스트 결합 검색"""
        results = []
        
        # 텍스트 기반
        if query_text:
            text_results = self.find_similar_by_text(query_text, top_k)
            problems = self.data_manager.get_all_problems()
            for idx, score in text_results:
                if idx < len(problems):
                    problems[idx]['similarity_score'] = score
                    problems[idx]['method'] = 'text_based'
                    results.append(problems[idx])
        
        return results[:top_k]


# ===== Gemini 통합 =====
class GeminiIntegration:
    """Gemini를 이용한 이미지 분석 및 답변"""
    
    @staticmethod
    def analyze_image_question(image: Image.Image, question: str, 
                                answer_style: str = "simple") -> Dict:
        """이미지 + 질문 분석"""
        if not Config.GEMINI_API_KEY:
            return {"error": "Gemini API 미설정"}
        
        try:
            model = genai.GenerativeModel(Config.GEMINI_MODEL)
            
            # 답변 스타일에 맞게 프롬프트 조정
            style_prompt = {
                'simple': "간단명료하게 1-2문장으로 답변하세요.",
                'detailed': "자세하고 단계별로 설명하세요. 과정을 포함하세요.",
                'step_by_step': "1단계, 2단계, 3단계 형식으로 단계별로 설명하세요.",
                'concept': "핵심 개념을 먼저 설명한 후 이 문제에 어떻게 적용되는지 설명하세요."
            }
            
            prompt = f"""{style_prompt.get(answer_style, style_prompt['simple'])}
            
사용자 질문: {question}

이미지를 분석하고 위 질문에 답변하세요."""
            
            response = model.generate_content([image, prompt])
            
            return {
                "status": "success",
                "answer": response.text,
                "style": answer_style,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            error_str = str(e)
            
            # 할당량 초과 오류 감지
            if "Quota exceeded" in error_str or "quota" in error_str.lower():
                return {
                    "status": "quota_exceeded",
                    "error_message": "❌ Gemini API 무료 할당량이 소진되었습니다. Google Cloud Console에서 결제 정보를 추가하거나 내일 다시 시도하세요."
                }
            elif "API_KEY" in error_str or "api_key" in error_str.lower():
                return {
                    "status": "invalid_api_key",
                    "error_message": "❌ API Key가 유효하지 않습니다. 설정 페이지에서 다시 확인하세요."
                }
            else:
                return {
                    "status": "error",
                    "error_message": f"⚠️ API 오류: {error_str[:100]}"
                }
    
    @staticmethod
    def summarize_learning_content(interactions: List[Dict]) -> str:
        """학습 상호작용을 단일 페이지로 요약"""
        if not Config.GEMINI_API_KEY:
            return "API 미설정"
        
        try:
            model = genai.GenerativeModel(Config.GEMINI_MODEL)
            
            interaction_text = "\n\n".join([
                f"Q: {i.get('question', '')}\nA: {i.get('answer', '')}"
                for i in interactions
            ])
            
            prompt = f"""다음 학습 내용을 한 페이지로 요약하세요. 마크다운 형식 사용:
- 핵심 개념
- 주요 공식/정리
- 실전 팁
- 복습 포인트

학습 내용:
{interaction_text}"""
            
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            error_str = str(e)
            
            if "Quota exceeded" in error_str or "quota" in error_str.lower():
                return """### ❌ API 할당량 초과

Gemini API 무료 할당량이 소진되었습니다.

**해결 방법:**
1. 내일 다시 시도 (할당량 자동 리셋)
2. [Google Cloud Console](https://console.cloud.google.com)에서 결제 정보 추가

**현재 사용 중인 모델:** `gemini-2.0-flash`"""
            else:
                return f"""### ⚠️ 요약 생성 실패

**오류:** {error_str[:200]}

설정에서 API Key를 다시 확인하세요."""


# ===== 복습 스케줄 (Spaced Repetition) =====
class ReviewScheduler:
    """복습 주기 관리 및 알림"""
    
    # 엥겔만의 간격 반복 곡선
    INTERVALS = [1, 3, 7, 14, 30]  # 일 단위
    
    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(exist_ok=True)
        self.review_history = self._load_history()
    
    def _load_history(self) -> Dict:
        """복습 기록 로드"""
        history_file = self.storage_dir / "review_history.json"
        if history_file.exists():
            with open(history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_history(self):
        """복습 기록 저장"""
        history_file = self.storage_dir / "review_history.json"
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(self.review_history, f, ensure_ascii=False, indent=2)
    
    def record_problem(self, problem_id: str, solved: bool = True):
        """문제 풀이 기록"""
        if problem_id not in self.review_history:
            self.review_history[problem_id] = {
                'attempts': 0,
                'correct': 0,
                'last_reviewed': None,
                'next_review': datetime.now().isoformat()
            }
        
        record = self.review_history[problem_id]
        record['attempts'] += 1
        if solved:
            record['correct'] += 1
        record['last_reviewed'] = datetime.now().isoformat()
        
        # 다음 복습 일시 계산
        correct_count = record['correct']
        if correct_count < len(self.INTERVALS):
            next_date = datetime.now() + timedelta(days=self.INTERVALS[correct_count])
            record['next_review'] = next_date.isoformat()
        
        self._save_history()
        return record
    
    def get_review_due(self) -> List[str]:
        """복습 기한이 된 문제들"""
        now = datetime.now()
        due = []
        
        for problem_id, record in self.review_history.items():
            next_review = datetime.fromisoformat(record['next_review'])
            if next_review <= now:
                due.append(problem_id)
        
        return due[:10]  # 최대 10개


# ===== Streamlit UI =====
def main():
    """메인 앱 인터페이스"""
    st.set_page_config(
        page_title="Qube 학습 MVP",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 모바일 반응형 스타일
    st.markdown("""
    <style>
    /* 모바일 최적화 */
    @media (max-width: 768px) {
        .main { padding: 0.5rem !important; }
        .block-container { padding: 0.5rem !important; max-width: 100% !important; }
        h1 { font-size: 1.5rem !important; }
        h2 { font-size: 1.2rem !important; }
        .stButton > button { width: 100%; padding: 0.5rem !important; font-size: 0.9rem; }
    }
    
    /* 일반 스타일 */
    .stButton > button {
        padding: 0.5rem 1rem !important;
        font-size: 1rem;
    }
    
    /* 파일 업로더 최적화 */
    .stFileUploader {
        width: 100% !important;
    }
    
    /* 입력 필드 */
    .stTextInput input, .stTextArea textarea {
        font-size: 1rem !important;
    }
    
    /* 메트릭 카드 */
    .stMetric {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    
    /* 익스팬더 */
    .streamlit-expanderHeader {
        font-size: 1rem !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("📚 Qube - AI 문제 기반 학습 플랫폼")
    st.markdown("*Gemini × YOLOv11 × SAM3 기반 맞춤형 학습*")
    
    # 시스템 상태 표시
    col1, col2 = st.columns(2)
    
    with col1:
        if not HAS_CV2:
            st.warning("⚠️ OpenCV가 설치되지 않았습니다. 이미지 특성 검색이 제한됩니다.")
        else:
            st.success("✓ OpenCV 설치됨")
    
    # 세션 상태 초기화
    if 'data_manager' not in st.session_state:
        st.session_state.data_manager = DataManager()
    
    with col2:
        problems_count = len(st.session_state.data_manager.get_all_problems())
        if problems_count > 0:
            st.success(f"✓ {problems_count}개 문제 로드됨")
        else:
            st.error("❌ 로드된 문제 없음")
    
    if 'review_scheduler' not in st.session_state:
        st.session_state.review_scheduler = ReviewScheduler(Config.OUTPUT_DIR)
    if 'similar_finder' not in st.session_state:
        st.session_state.similar_finder = SimilarProblemFinder(st.session_state.data_manager)
    if 'learning_history' not in st.session_state:
        st.session_state.learning_history = []
    
    # 사이드바 네비게이션
    with st.sidebar:
        st.markdown("## 🎯 기능 선택")
        
        mode = st.radio(
            "선택하세요",
            [
                "❓ 이미지 질문 답변",
                "🔍 유사 문제 검색",
                "📝 학습 콘텐츠 요약",
                "🔄 복습 스케줄",
                "⚙️ 설정"
            ],
            label_visibility="collapsed"
        )
        
        st.divider()
        st.caption("📱 모바일 팁: 상단 햄버거 메뉴로 네비게이션")
    
    # ===== 1) 이미지 질문 답변 =====
    if mode == "❓ 이미지 질문 답변":
        st.header("이미지 질문에 답변하기")
        st.write("이미지에서 문제를 보이고, 질문을 입력하면 Gemini가 답변해줍니다.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**이미지 선택** (JPG, PNG, WEBP 지원, 최대 200MB)")
            uploaded_image = st.file_uploader(
                "이미지 업로드",
                type=['jpg', 'jpeg', 'png', 'webp'],
                label_visibility="collapsed"
            )
        
        with col2:
            st.write("**답변 방식 선택**")
            answer_style = st.selectbox(
                "스타일",
                ["simple", "detailed", "step_by_step", "concept"],
                format_func=lambda x: {
                    "simple": "간단한 설명",
                    "detailed": "자세한 설명",
                    "step_by_step": "단계별 설명",
                    "concept": "개념 중심"
                }[x],
                label_visibility="collapsed"
            )
        
        question = st.text_area("📝 질문 입력", placeholder="무엇을 묻고 싶은가요?", height=100)
        
        if uploaded_image and question:
            col_a, col_b = st.columns([1, 1])
            
            with col_a:
                image = Image.open(uploaded_image)
                st.image(image, caption="📸 업로드된 이미지", use_container_width=True)
            
            with col_b:
                if st.button("🚀 답변 생성", use_container_width=True):
                    if not Config.GEMINI_API_KEY:
                        st.error("❌ API Key가 설정되지 않았습니다. 설정 페이지에서 API Key를 입력하세요.")
                    else:
                        with st.spinner("🤖 Gemini 분석 중..."):
                            result = GeminiIntegration.analyze_image_question(
                                image, question, answer_style
                            )
                        
                        if result.get('status') == 'success':
                            st.success("✅ 답변 완료!")
                            st.markdown(f"**답변:**\n\n{result['answer']}")
                            
                            # 학습 기록 저장
                            st.session_state.learning_history.append({
                                'timestamp': datetime.now().isoformat(),
                                'question': question,
                                'answer': result['answer'],
                                'style': answer_style
                            })
                            
                            # 유사 문제 검색
                            st.divider()
                            st.subheader("🔍 유사 문제")
                            with st.spinner("유사 문제 검색 중..."):
                                try:
                                    similar = st.session_state.similar_finder.find_similar_by_text(question)
                                except Exception as e:
                                    st.error(f"유사 문제 검색 실패: {e}")
                                    similar = []
                            
                            if similar:
                                for i, prob in enumerate(similar[:Config.MAX_SIMILAR_PROBLEMS], 1):
                                    with st.expander(f"📌 유사 문제 {i} - {prob['DomName']} (유사도: {prob['similarity']:.1%})"):
                                        st.write(f"**과목:** {prob['DomName']} / {prob['SubName']}")
                                        st.write(f"**난이도:** {prob['class']}")
                                        
                                        if prob['student_question']:
                                            st.write("**📝 학생 질문:**")
                                            st.write(prob['student_question'][:300] + "..." if len(prob['student_question']) > 300 else prob['student_question'])
                                        
                                        if prob['master_answer']:
                                            st.write("**💡 마스터 답변:**")
                                            st.write(prob['master_answer'][:500] + "..." if len(prob['master_answer']) > 500 else prob['master_answer'])
                            else:
                                st.info("유사한 문제를 찾지 못했습니다.")
                        else:
                            error_msg = result.get('error_message', '알 수 없는 오류 발생')
                            error_status = result.get('status', 'error')
                            
                            if error_status == 'quota_exceeded':
                                st.error(error_msg)
                                st.markdown("""
                                ---
                                ### 📋 해결 방법
                                
                                1. **내일 다시 시도** - 무료 할당량은 매일 자동 리셋됩니다
                                2. **유료 API 전환** (권장)
                                   - [Google Cloud Console](https://console.cloud.google.com) 접속
                                   - 결제 정보 추가
                                   - 프로젝트 설정에서 청구 활성화
                                   - 그러면 훨씬 더 높은 한도 사용 가능
                                
                                **현재 사용 중인 모델:** `gemini-2.0-flash`
                                """)
                            elif error_status == 'invalid_api_key':
                                st.error(error_msg)
                                st.info("⚙️ 설정 탭에서 API Key를 다시 입력해주세요.")
                            else:
                                st.error(f"❌ {error_msg}")
                                st.caption("설정 탭에서 API Key 상태를 확인하세요.")
                elif uploaded_image:
                    st.info("❓ 위에 질문을 입력하세요.")
                elif question:
                    st.info("📸 위에 이미지를 업로드하세요.")
    
    # ===== 2) 유사 문제 검색 =====
    elif mode == "🔍 유사 문제 검색":
        st.header("유사 문제 검색")
        
        search_type = st.radio("검색 방식", ["이미지 기반", "텍스트 기반"])
        
        if search_type == "이미지 기반":
            uploaded = st.file_uploader("이미지 선택", type=['jpg', 'jpeg', 'png'])
            if uploaded:
                image = Image.open(uploaded)
                st.image(image, use_container_width=True)
                
                if st.button("🔍 유사 문제 찾기"):
                    img_array = np.array(image)
                    results = st.session_state.similar_finder.find_similar_by_image(img_array)
                    st.json(results)
        
        else:
            query = st.text_area("검색어 입력")
            if query and st.button("검색"):
                with st.spinner("검색 중..."):
                    results = st.session_state.similar_finder.find_similar_by_text(query)
                
                if results:
                    st.success(f"✅ 찾은 문제: {len(results)}개")
                    for i, prob in enumerate(results, 1):
                        with st.expander(f"📌 {i}. {prob['DomName']} - {prob['SubName']} (유사도: {prob['similarity']:.1%})"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**과목 분류:** {prob['DomName']}")
                                st.write(f"**세부:** {prob['SubName']}")
                                st.write(f"**난이도:** {prob['class']}")
                            with col2:
                                st.write(f"**질문 ID:** {prob['QM_QST_NO']}")
                                st.write(f"**유사도:** {prob['similarity']:.1%}")
                            
                            if prob['student_question']:
                                st.write("**📝 학생 질문:**")
                                st.write(prob['student_question'][:200] + "..." if len(prob['student_question']) > 200 else prob['student_question'])
                            
                            if prob['master_answer']:
                                st.write("**💡 마스터 답변:**")
                                st.write(prob['master_answer'][:500] + "..." if len(prob['master_answer']) > 500 else prob['master_answer'])
                else:
                    st.warning("유사한 문제를 찾지 못했습니다.")
    
    # ===== 3) 학습 콘텐츠 요약 =====
    elif mode == "📝 학습 콘텐츠 요약":
        st.header("오늘의 학습 요약")
        
        if st.session_state.learning_history:
            st.write(f"📊 오늘 풀이한 문제: {len(st.session_state.learning_history)}개")
            
            if st.button("📄 요약 생성"):
                with st.spinner("요약 생성 중..."):
                    summary = GeminiIntegration.summarize_learning_content(
                        st.session_state.learning_history
                    )
                st.markdown(summary)
                
                # 저장 버튼
                if st.button("💾 요약 저장"):
                    filename = Config.OUTPUT_DIR / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                    filename.write_text(summary, encoding='utf-8')
                    st.success(f"✅ 저장됨: {filename.name}")
        else:
            st.info("아직 풀이한 문제가 없습니다.")
    
    # ===== 4) 복습 스케줄 =====
    elif mode == "🔄 복습 스케줄":
        st.header("복습 스케줄")
        
        scheduler = st.session_state.review_scheduler
        due_problems = scheduler.get_review_due()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📚 총 학습", len(scheduler.review_history))
        with col2:
            st.metric("🔄 복습 예정", len(due_problems))
        with col3:
            correct_count = sum(1 for r in scheduler.review_history.values() if r['correct'] > 0)
            st.metric("✅ 정답율", f"{correct_count}/{len(scheduler.review_history)}")
        
        if due_problems:
            st.warning("⏰ 복습이 필요한 문제들")
            for pid in due_problems:
                st.write(f"- {pid}")
        else:
            st.success("🎉 모든 복습이 완료되었습니다!")
    
    # ===== 5) 설정 =====
    elif mode == "⚙️ 설정":
        st.header("⚙️ 설정")
        
        st.subheader("🔑 API 설정")
        st.write("Gemini API 키를 설정하세요.")
        api_key = st.text_input(
            "API Key 입력",
            type="password",
            placeholder="sk-... 형태의 키를 입력하세요",
            help="[Google AI Studio](https://aistudio.google.com/app/apikey)에서 발급받으세요"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 저장"):
                if api_key:
                    os.environ["GEMINI_API_KEY_wj"] = api_key
                    Config.GEMINI_API_KEY = api_key
                    st.success("✅ API Key 저장됨")
                    st.rerun()
                else:
                    st.error("API Key를 입력하세요")
        
        with col2:
            if st.button("확인"):
                if Config.GEMINI_API_KEY:
                    try:
                        import google.generativeai as genai
                        genai.configure(api_key=Config.GEMINI_API_KEY)
                        model = genai.GenerativeModel(Config.GEMINI_MODEL)
                        st.success("✅ API Key 유효함")
                    except Exception as e:
                        st.error(f"❌ API 오류: {str(e)[:100]}")
                else:
                    st.error("❌ API Key가 설정되지 않았습니다")
        
        st.divider()
        st.subheader("📋 무료 API 할당량")
        
        st.markdown("""
        **Gemini API 무료 한도:**
        - 📊 일일 요청: 1,500개
        - ⏱️ 분당 요청: 15개
        - 🔤 분당 토큰: 10만 개
        
        **할당량 초과 시 해결 방법:**
        1. **내일 다시 시도** - 자동으로 리셋됨
        2. **유료 API 전환** (권장)
           - [Google Cloud Console](https://console.cloud.google.com) 접속
           - 결제 정보 추가 및 청구 활성화
           - 훨씬 더 높은 한도 사용 가능
        """)
        
        st.divider()
        st.subheader("🤖 모델 정보")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**사용 중인 모델:**")
            st.code(Config.GEMINI_MODEL)
        
        with col2:
            st.write(f"**API 상태:**")
            if Config.GEMINI_API_KEY:
                st.success("✓ API Key 설정됨")
            else:
                st.warning("⚠️ API Key 미설정 (위에서 설정하세요)")
        
        st.divider()
        st.subheader("📊 시스템 상태")
        
        problems = st.session_state.data_manager.get_all_problems()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📚 로드된 문제", f"{len(problems)}개")
        
        with col2:
            tfidf_status = "✓" if st.session_state.similar_finder.tfidf_matrix is not None else "✗"
            st.metric("📑 색인 상태", tfidf_status)
        
        with col3:
            cv2_status = "✓" if HAS_CV2 else "✗"
            st.metric("🖼️ OpenCV", cv2_status)
        
        st.divider()
        st.subheader("💾 학습 데이터")
        
        learning_count = len(st.session_state.learning_history)
        review_count = len(st.session_state.review_scheduler.review_history)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("✏️ 풀이한 문제", learning_count)
        with col2:
            st.metric("🔄 복습 기록", review_count)
        
        st.divider()
        st.subheader("📁 시스템 정보")
        
        with st.expander("🗂️ 데이터 경로"):
            st.code(str(Config.DATA_ROOT))
            st.code(str(Config.OUTPUT_DIR))
        
        # 데이터 상태 확인
        if len(problems) == 0:
            st.info("""
            ℹ️ **아직 마스터 데이터가 로드되지 않았습니다.**
            
            다음을 확인하세요:
            1. 데이터 경로가 존재하는가
            2. decoded_messages CSV 파일이 있는가
            3. 위에서 API Key를 설정했는가
            
            현재는 테스트용 샘플 데이터(5개)로 동작합니다.
            """)


if __name__ == "__main__":
    main()
