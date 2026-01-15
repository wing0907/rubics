# =========================================================
# Qube Agentic AI MVP - Streamlit UI
# - Data: qube_questions*.csv + (qube_messages*.csv or decoded_messages/*.csv)
# - Retrieval: TF-IDF
# - LLM optional (LLM_BASE_URL, LLM_API_KEY, LLM_MODEL env)
# - Streamlit: mobile friendly UI
# =========================================================

from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Optional, List, Dict, Any
from urllib.parse import unquote_plus

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------
# Settings
# -------------------------
st.set_page_config(page_title="Qube AI MVP", layout="centered")

DEFAULT_DATA_DIR = Path(os.environ.get("QUBE_DATA_DIR", r"D:\Users\Qube\0. 데이터\데이터분석_최종"))
DEFAULT_SERVER_IP = os.environ.get("MVP_SERVER_IP", "10.1.0.59")

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "").rstrip("/")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "60"))


# -------------------------
# Helpers
# -------------------------
def decode_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    try:
        return unquote_plus(s)
    except Exception:
        return s


def pick_text_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["qst_text_decoded", "qst_text_raw", "QstText_decoded", "QstText"]:
        if c in df.columns:
            return c
    return None


def load_questions_messages(data_dir: Path):
    data_dir = Path(data_dir)

    q_candidates = sorted(data_dir.glob("qube_questions*.csv"))
    if not q_candidates:
        raise FileNotFoundError("qube_questions*.csv not found in data_dir")

    df_q = pd.read_csv(q_candidates[0], encoding="utf-8-sig")
    q_name = q_candidates[0].name

    m_candidates = sorted(data_dir.glob("qube_messages*.csv"))
    if m_candidates:
        df_m = pd.read_csv(m_candidates[0], encoding="utf-8-sig")
        m_name = m_candidates[0].name
        return df_q, df_m, q_name, m_name

    msg_dir = data_dir / "decoded_messages"
    parts = sorted(msg_dir.glob("*.csv"))
    if not parts:
        raise FileNotFoundError("qube_messages*.csv not found and decoded_messages/*.csv not found")

    dfs = [pd.read_csv(p, encoding="utf-8-sig") for p in parts]
    df_m = pd.concat(dfs, ignore_index=True)
    return df_q, df_m, q_name, "decoded_messages folder"


def ensure_speaker_role(df_q: pd.DataFrame, df_m: pd.DataFrame) -> pd.DataFrame:
    m = df_m.copy()
    if "speaker_role" in m.columns:
        return m

    if "QM_QST_NO" not in m.columns:
        raise ValueError("df_messages must include QM_QST_NO")

    qmap = df_q.copy()

    if "student_key" not in qmap.columns:
        for cand in ["mem_key", "MemKey"]:
            if cand in qmap.columns:
                qmap["student_key"] = qmap[cand]
                break
    if "student_key" not in qmap.columns:
        qmap["student_key"] = ""

    if "RepMasterKey" not in qmap.columns:
        for cand in ["RepMemKey", "MasterKey"]:
            if cand in qmap.columns:
                qmap["RepMasterKey"] = qmap[cand]
                break
    if "RepMasterKey" not in qmap.columns:
        qmap["RepMasterKey"] = ""

    qmap = qmap[["QM_QST_NO", "student_key", "RepMasterKey"]].copy()
    qmap["QM_QST_NO"] = qmap["QM_QST_NO"].astype(str)

    m["QM_QST_NO"] = m["QM_QST_NO"].astype(str)
    m = m.merge(qmap, on="QM_QST_NO", how="left")

    msg_key_col = None
    for cand in ["msg_mem_key", "MemKey", "mem_key"]:
        if cand in m.columns:
            msg_key_col = cand
            break

    if not msg_key_col:
        m["speaker_role"] = "student"
        return m

    def role_row(r):
        mk = str(r.get(msg_key_col) or "")
        sk = str(r.get("student_key") or "")
        rk = str(r.get("RepMasterKey") or "")
        if sk and mk == sk:
            return "student"
        if rk and mk == rk:
            return "master"
        return "other"

    m["speaker_role"] = m.apply(role_row, axis=1)
    return m


def build_qa_pairs(df_m: pd.DataFrame) -> pd.DataFrame:
    m = df_m.copy()

    if "msg_idx" in m.columns:
        m["msg_idx"] = pd.to_numeric(m["msg_idx"], errors="coerce")

    sort_cols = []
    if "QM_QST_NO" in m.columns:
        sort_cols.append("QM_QST_NO")
    if "msg_idx" in m.columns:
        sort_cols.append("msg_idx")
    elif "msg_time" in m.columns:
        sort_cols.append("msg_time")

    if sort_cols:
        m = m.sort_values(sort_cols, kind="mergesort")

    txt_col = pick_text_col(m)
    if not txt_col:
        raise ValueError("message text column not found")

    for c in ["DomName", "SubName"]:
        if c not in m.columns:
            m[c] = ""

    rows = []
    for q_no, g in m.groupby("QM_QST_NO", dropna=False):
        if pd.isna(q_no) or str(q_no).strip() == "":
            continue

        dom = g["DomName"].iloc[0] if "DomName" in g.columns else ""
        sub = g["SubName"].iloc[0] if "SubName" in g.columns else ""

        student_msgs = g[g["speaker_role"] == "student"]
        master_msgs = g[g["speaker_role"] == "master"]

        q_text = "\n".join(student_msgs[txt_col].fillna("").astype(str).str.strip().tolist()).strip()
        a_text = "\n".join(master_msgs[txt_col].fillna("").astype(str).str.strip().tolist()).strip()

        if a_text == "":
            continue
        if len(q_text) < 5:
            continue

        rows.append(
            {
                "QM_QST_NO": str(q_no),
                "DomName": str(dom) if dom is not None else "",
                "SubName": str(sub) if sub is not None else "",
                "question_text": q_text,
                "answer_text": a_text,
            }
        )

    return pd.DataFrame(rows).reset_index(drop=True)


def build_tfidf_index(qa: pd.DataFrame):
    corpus = (qa["DomName"].fillna("") + " " + qa["SubName"].fillna("") + "\n" + qa["question_text"].fillna("")).tolist()
    vec = TfidfVectorizer(
        max_features=200000,
        ngram_range=(1, 2),
        min_df=2,
    )
    X = vec.fit_transform(corpus)
    return vec, X


def retrieve_similar(
    qa_df: pd.DataFrame,
    vec: TfidfVectorizer,
    X,
    question_text: str,
    dom: str,
    sub: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    query = f"{dom} {sub}\n{question_text}"
    qv = vec.transform([query])
    sims = cosine_similarity(qv, X).ravel()
    if sims.size == 0:
        return []

    idx = np.argsort(-sims)[:top_k]
    out = []
    for i in idx:
        out.append(
            {
                "score": float(sims[i]),
                "QM_QST_NO": qa_df.iloc[i]["QM_QST_NO"],
                "DomName": qa_df.iloc[i]["DomName"],
                "SubName": qa_df.iloc[i]["SubName"],
                "question_text": qa_df.iloc[i]["question_text"],
                "answer_text": qa_df.iloc[i]["answer_text"],
            }
        )
    return out


class LLMClient:
    def __init__(self):
        self.base_url = LLM_BASE_URL
        self.api_key = LLM_API_KEY
        self.model = LLM_MODEL
        self.timeout = LLM_TIMEOUT

    def enabled(self) -> bool:
        return bool(self.base_url) and bool(self.api_key)

    def chat(self, messages: List[Dict[str, Any]], temperature: float = 0.2, max_tokens: int = 900) -> str:
        if not self.enabled():
            raise RuntimeError("LLM not configured")

        import requests

        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        r = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]


def make_answer(
    qa_df: pd.DataFrame,
    vec: TfidfVectorizer,
    X,
    llm: LLMClient,
    question_text: str,
    dom: str = "",
    sub: str = "",
    image_urls: Optional[List[str]] = None,
    top_k: int = 4,
    sim_threshold: float = 0.18,
) -> Dict[str, Any]:
    q = decode_text(question_text).strip()
    dom = (dom or "").strip()
    sub = (sub or "").strip()
    image_urls = image_urls or []

    retrieved = retrieve_similar(qa_df, vec, X, q, dom, sub, top_k=top_k)
    best_score = retrieved[0]["score"] if retrieved else 0.0

    if (not llm.enabled()) or (best_score < sim_threshold):
        if retrieved:
            base = retrieved[0]
            draft = base["answer_text"].strip()
            if len(draft) > 1600:
                draft = draft[:1600] + "\n\n(중략)"
            return {
                "mode": "retrieval_only" if not llm.enabled() else "low_confidence_retrieval",
                "best_score": best_score,
                "answer": "유사 질문 기반 답변\n\n" + draft + "\n\n추가로 문제 지문, 보기, 사진을 더 주면 더 정확해짐",
                "citations": [{"QM_QST_NO": r["QM_QST_NO"], "score": r["score"]} for r in retrieved[:3]],
            }

        return {
            "mode": "no_data",
            "best_score": 0.0,
            "answer": "유사 사례가 부족함. 문제 지문, 보기, 사진을 추가로 주면 해결 가능",
            "citations": [],
        }

    exemplars = []
    for r in retrieved[:top_k]:
        exemplars.append(
            f"- 유사도 {r['score']:.3f} / Q {r['QM_QST_NO']} / {r['DomName']} {r['SubName']}\n"
            f"  유사 질문:\n{r['question_text'][:500]}\n"
            f"  마스터 답변:\n{r['answer_text'][:900]}\n"
        )
    exemplar_block = "\n".join(exemplars)

    system = (
        "너는 메가스터디 큐브의 AI 마스터 역할.\n"
        "학생 질문에 빠르고 정확하게 풀이 흐름을 단계적으로 설명.\n"
        "근거 부족하면 추가 정보 요청.\n"
        "사실을 만들어내지 말기.\n"
    )

    user_text = (
        f"과목: {dom} / {sub}\n"
        f"학생 질문:\n{q}\n\n"
        f"참고 가능한 유사 사례(마스터 답변 포함):\n{exemplar_block}\n\n"
        "출력 형식:\n"
        "결론 한 줄\n"
        "풀이 단계 3~8개\n"
        "실수 포인트 1~2개\n"
    )

    if image_urls:
        user_text += "\n\n첨부 이미지 URL:\n" + "\n".join(image_urls)

    messages = [{"role": "system", "content": system}, {"role": "user", "content": user_text}]
    ans = llm.chat(messages, temperature=0.2, max_tokens=900)

    return {
        "mode": "llm_with_retrieval",
        "best_score": best_score,
        "answer": ans,
        "citations": [{"QM_QST_NO": r["QM_QST_NO"], "score": r["score"]} for r in retrieved[:3]],
    }


def get_local_ipv4_candidates() -> List[str]:
    ips = set()
    try:
        hostname = socket.gethostname()
        ips.add(socket.gethostbyname(hostname))
    except Exception:
        pass

    try:
        for info in socket.getaddrinfo(socket.gethostname(), None):
            addr = info[4][0]
            if "." in addr and not addr.startswith("127."):
                ips.add(addr)
    except Exception:
        pass

    clean = []
    for ip in ips:
        if ip and ip != "127.0.0.1" and "." in ip:
            clean.append(ip)
    return sorted(clean)


# -------------------------
# Cache heavy steps
# -------------------------
@st.cache_data(show_spinner=False)
def cached_load(data_dir: str):
    df_q, df_m, q_src, m_src = load_questions_messages(Path(data_dir))
    df_m = ensure_speaker_role(df_q, df_m)
    qa_df = build_qa_pairs(df_m)
    return df_q, df_m, qa_df, q_src, m_src


@st.cache_resource(show_spinner=False)
def cached_index(qa_df: pd.DataFrame):
    return build_tfidf_index(qa_df)


# -------------------------
# UI
# -------------------------
st.title("Qube AI MVP")
st.caption("폰에서 확인하려면 Streamlit을 0.0.0.0로 바인딩하고 포트를 열어야 함")

with st.sidebar:
    st.subheader("실행 정보")
    st.write("LLM enabled:", bool(LLM_BASE_URL and LLM_API_KEY))
    st.write("Suggested server ip:", DEFAULT_SERVER_IP)
    st.write("Local IPv4 candidates:", ", ".join(get_local_ipv4_candidates()) or "not found")

    st.subheader("데이터 경로")
    data_dir = st.text_input("QUBE_DATA_DIR", value=str(DEFAULT_DATA_DIR))
    reload_btn = st.button("데이터 다시 로드")

if reload_btn:
    st.cache_data.clear()
    st.cache_resource.clear()

try:
    with st.spinner("데이터 로딩"):
        df_questions, df_messages, qa_df, q_src, m_src = cached_load(data_dir)

    vec, X = cached_index(qa_df)
    llm = LLMClient()

    st.success(f"로딩 완료  questions {df_questions.shape}  messages {df_messages.shape}  qa {qa_df.shape}")
    st.caption(f"sources  questions {q_src}  messages {m_src}")

except Exception as e:
    st.error(f"데이터 로딩 실패: {e}")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    dom = st.text_input("DomName", value="")
with col2:
    sub = st.text_input("SubName", value="")

q = st.text_area("질문", height=170)
image_urls_text = st.text_area("이미지 URL 목록 (한 줄에 하나, 선택)", height=80)

image_urls = [x.strip() for x in image_urls_text.splitlines() if x.strip()]
top_k = st.slider("유사 사례 top_k", min_value=1, max_value=10, value=4)
sim_threshold = st.slider("LLM 전환 임계값", min_value=0.05, max_value=0.60, value=0.18, step=0.01)

ask = st.button("질문 보내기")

if ask:
    if not q.strip():
        st.warning("질문이 비어있음")
    else:
        with st.spinner("답변 생성"):
            out = make_answer(
                qa_df=qa_df,
                vec=vec,
                X=X,
                llm=llm,
                question_text=q,
                dom=dom,
                sub=sub,
                image_urls=image_urls,
                top_k=top_k,
                sim_threshold=sim_threshold,
            )

        st.subheader("답변")
        st.write(out.get("answer", ""))

        st.subheader("메타")
        st.write("mode:", out.get("mode"))
        st.write("best_score:", float(out.get("best_score", 0.0)))

        cits = out.get("citations", [])
        if cits:
            st.subheader("유사 사례")
            st.dataframe(pd.DataFrame(cits))

        st.divider()
        st.caption("폰에서 접속은 http 사용  https 아님")


# Optional QR inside UI
try:
    import qrcode
    from PIL import Image
    import io

    st.subheader("QR로 폰 접속")
    server_ip_for_qr = st.text_input("QR에 넣을 서버 IP", value=DEFAULT_SERVER_IP)
    server_port_for_qr = st.text_input("서버 포트", value="8501")
    url = f"http://{server_ip_for_qr}:{server_port_for_qr}"

    img = qrcode.make(url)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    st.image(buf.getvalue(), caption=url, use_container_width=False)

except Exception:
    st.caption("QR 기능은 qrcode pillow 설치되면 활성화")

