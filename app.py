from __future__ import annotations

import html
import os
import re
import threading
from pathlib import Path
from typing import Any

# Menjaga penggunaan resource tetap ringan di Streamlit Community Cloud.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. KONFIGURASI APLIKASI
# ============================================================
APP_DIR = Path(__file__).resolve().parent
CHUNKS_PATH = APP_DIR / "bbc_news_chunks.parquet"
EMBEDDINGS_PATH = APP_DIR / "bbc_chunk_embeddings.npy"

# Harus sama dengan model ketika embedding korpus dibuat.
EMBEDDING_MODEL_NAME = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

TOP_K = 3
MIN_CHUNK_SCORE = 0.50
MIN_SENTENCE_SCORE = 0.48
MAX_CANDIDATE_SENTENCES = 24
SEARCH_BLOCK_SIZE = 4096

NO_ANSWER_MESSAGE = (
    "Maaf, saya tidak dapat menjawab pertanyaan tersebut karena "
    "dokumen yang tersedia tidak memuat sumber yang cukup relevan."
)

LIVE_INFORMATION_MARKERS = (
    "hari ini",
    "sekarang",
    "saat ini",
    "terbaru",
    "terkini",
    "cuaca",
    "prakiraan",
    "ramalan cuaca",
    "suhu",
    "today",
    "right now",
    "currently",
    "latest",
    "weather",
    "forecast",
    "temperature",
)

REQUIRED_COLUMNS = {
    "doc_id",
    "chunk_id",
    "category",
    "chunk_text",
}


# ============================================================
# 2. HALAMAN DAN DESAIN RESPONSIF
# ============================================================
st.set_page_config(
    page_title="BBC News RAG",
    page_icon="🗞️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        :root {
            --page: #f4efe6;
            --page-deep: #ebe2d4;
            --surface: #fffdf9;
            --surface-soft: #f9f5ee;
            --ink: #17201e;
            --ink-soft: #53615d;
            --line: rgba(23, 32, 30, 0.12);
            --brand: #cf3f2f;
            --brand-dark: #9f2d22;
            --green: #21463d;
            --green-soft: #e6efe9;
            --amber: #bd7417;
            --shadow: 0 18px 55px rgba(49, 39, 26, 0.09);
            --shadow-soft: 0 8px 24px rgba(49, 39, 26, 0.07);
            --radius-xl: 26px;
            --radius-lg: 20px;
            --radius-md: 15px;
        }

        html,
        body,
        [class*="css"] {
            font-family:
                "Avenir Next",
                "Trebuchet MS",
                "Segoe UI",
                sans-serif;
        }

        html {
            color-scheme: light !important;
            background: var(--page) !important;
        }

        body,
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"] {
            background:
                radial-gradient(
                    circle at 92% 5%,
                    rgba(207, 63, 47, 0.12),
                    transparent 28rem
                ),
                radial-gradient(
                    circle at 8% 92%,
                    rgba(33, 70, 61, 0.10),
                    transparent 30rem
                ),
                linear-gradient(
                    180deg,
                    #faf7f1 0%,
                    var(--page) 58%,
                    var(--page-deep) 100%
                ) !important;
            color: var(--ink) !important;
        }

        [data-testid="stAppViewContainer"] {
            min-height: 100svh;
        }

        [data-testid="stHeader"] {
            background: transparent !important;
        }

        #MainMenu,
        footer {
            visibility: hidden;
        }

        .block-container {
            width: min(100%, 920px);
            max-width: 920px;
            padding-top: 1.55rem;
            padding-left: 1.25rem;
            padding-right: 1.25rem;
            padding-bottom: 8.75rem;
        }

        /* ---------- HERO ---------- */
        .rag-hero {
            position: relative;
            overflow: hidden;
            margin: 0 0 1.25rem;
            padding: 1.8rem 1.9rem 1.65rem;
            border: 1px solid rgba(255, 255, 255, 0.72);
            border-radius: var(--radius-xl);
            background:
                linear-gradient(
                    135deg,
                    rgba(255, 255, 255, 0.96),
                    rgba(255, 251, 245, 0.88)
                );
            box-shadow: var(--shadow);
            isolation: isolate;
        }

        .rag-hero::after {
            content: "";
            position: absolute;
            z-index: -1;
            width: 12rem;
            height: 12rem;
            top: -5.3rem;
            right: -3rem;
            border-radius: 50%;
            background:
                radial-gradient(
                    circle,
                    rgba(207, 63, 47, 0.28),
                    rgba(207, 63, 47, 0)
                );
        }

        .rag-brand-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 1.1rem;
        }

        .rag-brand {
            display: inline-flex;
            align-items: center;
            gap: 0.65rem;
            color: var(--green);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }

        .rag-brand-mark {
            display: grid;
            place-items: center;
            width: 2.15rem;
            height: 2.15rem;
            border-radius: 11px;
            color: #ffffff;
            background: var(--brand);
            box-shadow: 0 8px 18px rgba(207, 63, 47, 0.24);
            font-size: 1rem;
        }

        .rag-status {
            display: inline-flex;
            align-items: center;
            gap: 0.42rem;
            flex: 0 0 auto;
            padding: 0.45rem 0.7rem;
            border-radius: 999px;
            color: var(--green);
            background: var(--green-soft);
            font-size: 0.72rem;
            font-weight: 800;
        }

        .rag-status-dot {
            width: 0.48rem;
            height: 0.48rem;
            border-radius: 50%;
            background: #2c8a68;
            box-shadow: 0 0 0 4px rgba(44, 138, 104, 0.12);
        }

        .rag-title {
            max-width: 690px;
            margin: 0;
            color: var(--ink) !important;
            font-family:
                "Iowan Old Style",
                "Palatino Linotype",
                Georgia,
                serif;
            font-size: clamp(2rem, 5vw, 3.35rem);
            line-height: 0.98;
            letter-spacing: -0.048em;
        }

        .rag-title em {
            color: var(--brand);
            font-style: italic;
        }

        .rag-subtitle {
            max-width: 700px;
            margin: 1rem 0 0;
            color: var(--ink-soft) !important;
            font-size: 1rem;
            line-height: 1.65;
        }

        .rag-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 1.2rem;
        }

        .rag-chip {
            padding: 0.48rem 0.72rem;
            border: 1px solid var(--line);
            border-radius: 999px;
            color: var(--ink-soft) !important;
            background: rgba(255, 255, 255, 0.72);
            font-size: 0.76rem;
            font-weight: 700;
        }

        /* ---------- EMPTY STATE ---------- */
        .empty-state {
            margin: 0.35rem 0 1rem;
            padding: 1.15rem 1.2rem;
            border: 1px dashed rgba(23, 32, 30, 0.20);
            border-radius: var(--radius-lg);
            color: var(--ink-soft) !important;
            background: rgba(255, 253, 249, 0.58);
        }

        .empty-title {
            margin-bottom: 0.65rem;
            color: var(--ink) !important;
            font-family:
                "Iowan Old Style",
                "Palatino Linotype",
                Georgia,
                serif;
            font-size: 1.05rem;
            font-weight: 700;
        }

        .empty-examples {
            display: grid;
            gap: 0.5rem;
        }

        .empty-example {
            display: flex;
            align-items: flex-start;
            gap: 0.65rem;
            color: var(--ink-soft) !important;
            font-size: 0.88rem;
            line-height: 1.45;
        }

        .empty-number {
            display: grid;
            place-items: center;
            min-width: 1.45rem;
            height: 1.45rem;
            border-radius: 50%;
            color: #ffffff !important;
            background: var(--green);
            font-size: 0.68rem;
            font-weight: 800;
        }

        /* ---------- CHAT MESSAGE ---------- */
        [data-testid="stChatMessage"] {
            min-height: 0 !important;
            height: auto !important;
            margin: 0 0 0.8rem !important;
            padding: 1rem 1.1rem !important;
            gap: 0.85rem !important;
            border: 1px solid var(--line) !important;
            border-radius: var(--radius-lg) !important;
            background: rgba(255, 253, 249, 0.94) !important;
            box-shadow: var(--shadow-soft) !important;
            overflow: visible !important;
        }

        [data-testid="stChatMessage"] > div {
            min-width: 0 !important;
        }

        [data-testid="stChatMessageAvatarUser"],
        [data-testid="stChatMessageAvatarAssistant"] {
            width: 2.45rem !important;
            height: 2.45rem !important;
            min-width: 2.45rem !important;
            border-radius: 13px !important;
            box-shadow: 0 7px 16px rgba(40, 30, 20, 0.12);
        }

        [data-testid="stChatMessageAvatarUser"] {
            background: var(--brand) !important;
        }

        [data-testid="stChatMessageAvatarAssistant"] {
            background: var(--green) !important;
        }

        [data-testid="stChatMessageContent"],
        [data-testid="stChatMessageContent"] *,
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li,
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] span {
            color: var(--ink) !important;
            -webkit-text-fill-color: var(--ink) !important;
            opacity: 1 !important;
        }

        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
            min-height: 0 !important;
            font-size: 0.98rem;
            line-height: 1.62;
            overflow-wrap: anywhere;
            word-break: normal;
        }

        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
            margin-bottom: 0.65rem;
        }

        [data-testid="stChatMessage"] blockquote {
            margin: 0.7rem 0 !important;
            padding: 0.8rem 0.95rem !important;
            border-left: 3px solid var(--brand) !important;
            border-radius: 0 12px 12px 0;
            color: #37433f !important;
            background: var(--surface-soft) !important;
        }

        [data-testid="stChatMessage"] blockquote p {
            margin: 0 !important;
            color: #37433f !important;
            -webkit-text-fill-color: #37433f !important;
        }

        [data-testid="stChatMessage"] strong {
            color: var(--green) !important;
            -webkit-text-fill-color: var(--green) !important;
        }

        /* ---------- SCORE BADGES ---------- */
        .score-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin: 0.35rem 0 0.2rem;
        }

        .score-chip {
            display: inline-flex;
            align-items: center;
            padding: 0.34rem 0.58rem;
            border: 1px solid rgba(33, 70, 61, 0.13);
            border-radius: 999px;
            color: var(--green) !important;
            background: var(--green-soft);
            font-size: 0.72rem;
            font-weight: 800;
        }

        /* ---------- EXPANDER & SOURCE CARDS ---------- */
        [data-testid="stExpander"] {
            margin-top: 0.7rem;
            border: 1px solid var(--line) !important;
            border-radius: var(--radius-md) !important;
            background: rgba(249, 245, 238, 0.75) !important;
            overflow: hidden;
        }

        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary * {
            color: var(--ink) !important;
            -webkit-text-fill-color: var(--ink) !important;
            font-weight: 750 !important;
        }

        .source-card {
            margin: 0.65rem 0;
            padding: 0.9rem;
            border: 1px solid var(--line);
            border-radius: 14px;
            background: var(--surface);
        }

        .source-head {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 0.8rem;
            margin-bottom: 0.55rem;
        }

        .source-id {
            min-width: 0;
            color: var(--green) !important;
            font-size: 0.81rem;
            font-weight: 850;
            overflow-wrap: anywhere;
        }

        .source-badge {
            flex: 0 0 auto;
            padding: 0.26rem 0.5rem;
            border-radius: 999px;
            color: #ffffff !important;
            background: var(--brand);
            font-size: 0.65rem;
            font-weight: 850;
        }

        .source-badge.candidate {
            color: var(--green) !important;
            background: var(--green-soft);
        }

        .source-meta {
            margin-bottom: 0.5rem;
            color: var(--amber) !important;
            font-size: 0.72rem;
            font-weight: 800;
        }

        .source-text {
            color: var(--ink-soft) !important;
            font-size: 0.84rem;
            line-height: 1.55;
            overflow-wrap: anywhere;
        }

        /* ---------- CHAT INPUT ---------- */
        [data-testid="stBottom"] {
            padding-bottom: env(safe-area-inset-bottom);
            background:
                linear-gradient(
                    180deg,
                    rgba(244, 239, 230, 0),
                    rgba(244, 239, 230, 0.92) 34%,
                    rgba(244, 239, 230, 1) 100%
                ) !important;
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
        }

        [data-testid="stBottom"] > div {
            width: min(calc(100% - 2.5rem), 895px) !important;
            margin: 0 auto !important;
            padding-bottom: 0.9rem !important;
            background: transparent !important;
        }

        [data-testid="stChatInput"] {
            min-height: 3.65rem !important;
            border: 1px solid rgba(23, 32, 30, 0.15) !important;
            border-radius: 20px !important;
            background: rgba(255, 253, 249, 0.98) !important;
            box-shadow: 0 16px 42px rgba(43, 33, 23, 0.16) !important;
            overflow: hidden;
        }

        [data-testid="stChatInput"] > div,
        [data-testid="stChatInput"] textarea {
            background: transparent !important;
        }

        [data-testid="stChatInput"] textarea {
            color: var(--ink) !important;
            -webkit-text-fill-color: var(--ink) !important;
            caret-color: var(--brand) !important;
            font-size: 0.98rem !important;
            line-height: 1.45 !important;
        }

        [data-testid="stChatInput"] textarea::placeholder {
            color: #7b8581 !important;
            -webkit-text-fill-color: #7b8581 !important;
            opacity: 1 !important;
        }

        [data-testid="stChatInput"] button {
            width: 2.45rem !important;
            height: 2.45rem !important;
            border-radius: 13px !important;
            color: #ffffff !important;
            background: var(--brand) !important;
        }

        [data-testid="stChatInput"] button:hover {
            background: var(--brand-dark) !important;
        }

        /* ---------- SIDEBAR ---------- */
        [data-testid="stSidebar"] {
            background: var(--green) !important;
        }

        [data-testid="stSidebar"] *,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span {
            color: #f8f2e9 !important;
            -webkit-text-fill-color: #f8f2e9 !important;
        }

        [data-testid="stSidebar"] button {
            border-color: rgba(255, 255, 255, 0.28) !important;
            background: rgba(255, 255, 255, 0.10) !important;
        }

        /* ---------- RESPONSIVE ---------- */
        @media (max-width: 700px) {
            .block-container {
                width: 100%;
                padding-top: 0.7rem;
                padding-left: 0.72rem;
                padding-right: 0.72rem;
                padding-bottom:
                    calc(8.4rem + env(safe-area-inset-bottom));
            }

            .rag-hero {
                margin-bottom: 0.85rem;
                padding: 1.15rem 1.05rem 1.1rem;
                border-radius: 20px;
            }

            .rag-brand-row {
                margin-bottom: 0.85rem;
            }

            .rag-brand {
                gap: 0.5rem;
                font-size: 0.69rem;
                letter-spacing: 0.09em;
            }

            .rag-brand-mark {
                width: 1.9rem;
                height: 1.9rem;
                border-radius: 10px;
                font-size: 0.88rem;
            }

            .rag-status {
                padding: 0.36rem 0.55rem;
                font-size: 0.64rem;
            }

            .rag-title {
                font-size: clamp(1.8rem, 9vw, 2.45rem);
                line-height: 1.02;
            }

            .rag-subtitle {
                margin-top: 0.75rem;
                font-size: 0.88rem;
                line-height: 1.52;
            }

            .rag-chips {
                gap: 0.4rem;
                margin-top: 0.9rem;
            }

            .rag-chip {
                padding: 0.4rem 0.58rem;
                font-size: 0.66rem;
            }

            .empty-state {
                padding: 0.95rem;
                border-radius: 16px;
            }

            .empty-title {
                font-size: 0.96rem;
            }

            .empty-example {
                font-size: 0.8rem;
            }

            [data-testid="stChatMessage"] {
                width: 100% !important;
                min-height: 0 !important;
                margin-bottom: 0.62rem !important;
                padding: 0.78rem 0.82rem !important;
                gap: 0.62rem !important;
                border-radius: 17px !important;
            }

            [data-testid="stChatMessageAvatarUser"],
            [data-testid="stChatMessageAvatarAssistant"] {
                width: 2.05rem !important;
                height: 2.05rem !important;
                min-width: 2.05rem !important;
                border-radius: 11px !important;
            }

            [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
                width: 100% !important;
                min-height: 0 !important;
                font-size: 0.88rem !important;
                line-height: 1.55 !important;
            }

            [data-testid="stChatMessage"] blockquote {
                padding: 0.68rem 0.72rem !important;
            }

            .score-row {
                gap: 0.34rem;
            }

            .score-chip {
                padding: 0.28rem 0.46rem;
                font-size: 0.64rem;
            }

            .source-card {
                padding: 0.72rem;
            }

            .source-head {
                display: block;
            }

            .source-badge {
                display: inline-flex;
                margin-top: 0.45rem;
            }

            .source-text {
                font-size: 0.78rem;
            }

            [data-testid="stBottom"] > div {
                width: calc(100% - 1rem) !important;
                padding-bottom:
                    calc(0.45rem + env(safe-area-inset-bottom)) !important;
            }

            [data-testid="stChatInput"] {
                min-height: 3.35rem !important;
                border-radius: 17px !important;
            }

            [data-testid="stChatInput"] textarea {
                font-size: 0.9rem !important;
            }

            [data-testid="stChatInput"] button {
                width: 2.2rem !important;
                height: 2.2rem !important;
                border-radius: 11px !important;
            }
        }

        @media (max-width: 380px) {
            .rag-status {
                display: none;
            }

            .rag-chips {
                display: none;
            }

            .block-container {
                padding-left: 0.55rem;
                padding-right: 0.55rem;
            }

            [data-testid="stChatMessage"] {
                padding: 0.7rem !important;
            }
        }

        @media (prefers-reduced-motion: no-preference) {
            .rag-hero {
                animation: rise-in 420ms ease-out both;
            }

            [data-testid="stChatMessage"] {
                animation: rise-in 260ms ease-out both;
            }

            @keyframes rise-in {
                from {
                    opacity: 0;
                    transform: translateY(8px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. DATA DAN MODEL
# ============================================================
def validate_required_files() -> None:
    missing = [
        path.name
        for path in (CHUNKS_PATH, EMBEDDINGS_PATH)
        if not path.exists()
    ]

    if missing:
        st.error(
            "File knowledge base belum ditemukan: "
            + ", ".join(missing)
            + ". Letakkan file tersebut pada folder yang sama dengan app.py."
        )
        st.stop()


@st.cache_resource(show_spinner=False)
def load_knowledge_base() -> tuple[pd.DataFrame, np.ndarray]:
    chunks = pd.read_parquet(
        CHUNKS_PATH,
        columns=[
            "doc_id",
            "chunk_id",
            "category",
            "chunk_text",
        ],
    ).reset_index(drop=True)

    missing_columns = REQUIRED_COLUMNS.difference(chunks.columns)
    if missing_columns:
        raise ValueError(
            "Kolom pada file parquet belum lengkap: "
            + ", ".join(sorted(missing_columns))
        )

    embeddings = np.load(
        EMBEDDINGS_PATH,
        mmap_mode="r",
        allow_pickle=False,
    )

    if embeddings.ndim != 2:
        raise ValueError("Embedding harus berupa matriks dua dimensi.")

    if len(chunks) != embeddings.shape[0]:
        raise ValueError(
            "Jumlah chunk dan embedding berbeda. "
            f"Chunk={len(chunks)}, embedding={embeddings.shape[0]}."
        )

    return chunks, embeddings


@st.cache_resource(show_spinner="Memuat mesin pencarian...")
def load_embedder() -> tuple[SentenceTransformer, threading.Lock]:
    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME,
        device="cpu",
    )
    model.eval()
    return model, threading.Lock()


# ============================================================
# 4. RETRIEVAL DAN VALIDASI
# ============================================================
def normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", question).strip()


def asks_for_live_information(question: str) -> bool:
    normalized = normalize_question(question).lower()
    return any(
        marker in normalized
        for marker in LIVE_INFORMATION_MARKERS
    )


def encode_query(
    question: str,
    embedder: SentenceTransformer,
    inference_lock: threading.Lock,
) -> np.ndarray:
    with inference_lock:
        if hasattr(embedder, "encode_query"):
            vector = embedder.encode_query(
                [question],
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0]
        else:
            vector = embedder.encode(
                [question],
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0]

    return np.asarray(vector, dtype=np.float32)


def cosine_scores_in_blocks(
    corpus_embeddings: np.ndarray,
    query_vector: np.ndarray,
    block_size: int = SEARCH_BLOCK_SIZE,
) -> np.ndarray:
    total_rows = corpus_embeddings.shape[0]
    scores = np.empty(total_rows, dtype=np.float32)

    query_norm = float(np.linalg.norm(query_vector))
    if query_norm == 0.0:
        return np.full(total_rows, -1.0, dtype=np.float32)

    normalized_query = query_vector / query_norm

    for start in range(0, total_rows, block_size):
        end = min(start + block_size, total_rows)
        block = np.asarray(
            corpus_embeddings[start:end],
            dtype=np.float32,
        )
        norms = np.linalg.norm(block, axis=1)
        scores[start:end] = (
            block @ normalized_query
        ) / np.maximum(norms, 1e-12)

    return scores


def retrieve_documents(
    question: str,
    chunks: pd.DataFrame,
    corpus_embeddings: np.ndarray,
    embedder: SentenceTransformer,
    inference_lock: threading.Lock,
    top_k: int = TOP_K,
) -> tuple[pd.DataFrame, np.ndarray]:
    query_vector = encode_query(
        question,
        embedder,
        inference_lock,
    )

    scores = cosine_scores_in_blocks(
        corpus_embeddings,
        query_vector,
    )

    top_k = min(top_k, len(scores))
    if top_k <= 0:
        return pd.DataFrame(), query_vector

    top_indices = np.argpartition(
        scores,
        -top_k,
    )[-top_k:]

    top_indices = top_indices[
        np.argsort(scores[top_indices])[::-1]
    ]

    rows: list[dict[str, Any]] = []

    for rank, index in enumerate(top_indices, start=1):
        row = chunks.iloc[int(index)]
        rows.append(
            {
                "rank": rank,
                "doc_id": str(row["doc_id"]),
                "chunk_id": str(row["chunk_id"]),
                "category": str(row["category"]),
                "score": float(scores[index]),
                "text": str(row["chunk_text"]),
            }
        )

    return pd.DataFrame(rows), query_vector


def split_into_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", str(text)).strip()
    if not text:
        return []

    raw_sentences = re.split(
        r"(?<=[.!?])\s+(?=[A-Z0-9\"'])",
        text,
    )

    output: list[str] = []

    for sentence in raw_sentences:
        sentence = sentence.strip()
        if len(sentence.split()) < 5:
            continue

        words = sentence.split()

        if len(words) <= 70:
            output.append(sentence)
            continue

        window_size = 55
        overlap = 10
        step = window_size - overlap

        for start in range(0, len(words), step):
            window = words[start:start + window_size]
            if len(window) < 8:
                break
            output.append(" ".join(window))

    return output


def build_sentence_candidates(
    retrieved: pd.DataFrame,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    for row in retrieved.itertuples():
        for sentence in split_into_sentences(row.text):
            candidates.append(
                {
                    "sentence": sentence,
                    "doc_id": row.doc_id,
                    "chunk_id": row.chunk_id,
                    "category": row.category,
                    "chunk_score": float(row.score),
                }
            )

            if len(candidates) >= MAX_CANDIDATE_SENTENCES:
                return candidates

    return candidates


def rank_candidate_sentences(
    candidates: list[dict[str, Any]],
    query_vector: np.ndarray,
    embedder: SentenceTransformer,
    inference_lock: threading.Lock,
) -> list[dict[str, Any]]:
    if not candidates:
        return []

    texts = [item["sentence"] for item in candidates]

    with inference_lock:
        if hasattr(embedder, "encode_document"):
            sentence_embeddings = embedder.encode_document(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=16,
            )
        else:
            sentence_embeddings = embedder.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=16,
            )

    sentence_embeddings = np.asarray(
        sentence_embeddings,
        dtype=np.float32,
    )

    normalized_query = query_vector / max(
        float(np.linalg.norm(query_vector)),
        1e-12,
    )

    semantic_scores = sentence_embeddings @ normalized_query

    ranked: list[dict[str, Any]] = []

    for item, score in zip(candidates, semantic_scores):
        ranked_item = dict(item)
        ranked_item["sentence_score"] = float(score)
        ranked.append(ranked_item)

    ranked.sort(
        key=lambda item: (
            item["sentence_score"],
            item["chunk_score"],
        ),
        reverse=True,
    )

    return ranked


def clean_excerpt(text: str, max_characters: int = 520) -> str:
    text = re.sub(r"\s+", " ", text).strip()

    if text:
        text = text[0].upper() + text[1:]

    if len(text) <= max_characters:
        return text

    shortened = text[:max_characters].rsplit(" ", 1)[0]
    return shortened.rstrip(" ,;:-") + "…"


def create_extractive_answer(
    ranked_sentences: list[dict[str, Any]],
) -> tuple[str, list[str], list[str]]:
    best = ranked_sentences[0]
    selected = [best]

    for candidate in ranked_sentences[1:]:
        if len(selected) >= 2:
            break

        if candidate["sentence"] == best["sentence"]:
            continue

        if candidate["sentence_score"] < best["sentence_score"] - 0.035:
            continue

        selected.append(candidate)

    excerpts = [
        clean_excerpt(item["sentence"])
        for item in selected
    ]

    source_labels: list[str] = []
    used_chunk_ids: list[str] = []

    for item in selected:
        label = f"{item['doc_id']} · {item['chunk_id']}"
        if label not in source_labels:
            source_labels.append(label)
        if item["chunk_id"] not in used_chunk_ids:
            used_chunk_ids.append(item["chunk_id"])

    quoted_text = "\n\n".join(
        f"> {excerpt}"
        for excerpt in excerpts
    )

    answer = (
        "Berdasarkan sumber yang tersedia, informasi yang paling "
        "relevan adalah:\n\n"
        f"{quoted_text}\n\n"
        f"**Sumber:** {', '.join(source_labels)}"
    )

    return answer, source_labels, used_chunk_ids


def answer_question(
    question: str,
    chunks: pd.DataFrame,
    corpus_embeddings: np.ndarray,
    embedder: SentenceTransformer,
    inference_lock: threading.Lock,
) -> dict[str, Any]:
    question = normalize_question(question)

    if asks_for_live_information(question):
        return {
            "answer": NO_ANSWER_MESSAGE,
            "supported": False,
            "reason": (
                "Pertanyaan meminta informasi real-time, sedangkan "
                "knowledge base hanya berisi dokumen berita statis."
            ),
            "top_score": None,
            "sentence_score": None,
            "retrieved": [],
            "used_chunk_ids": [],
        }

    retrieved, query_vector = retrieve_documents(
        question=question,
        chunks=chunks,
        corpus_embeddings=corpus_embeddings,
        embedder=embedder,
        inference_lock=inference_lock,
    )

    if retrieved.empty:
        return {
            "answer": NO_ANSWER_MESSAGE,
            "supported": False,
            "reason": "Tidak ada dokumen yang berhasil diambil.",
            "top_score": None,
            "sentence_score": None,
            "retrieved": [],
            "used_chunk_ids": [],
        }

    top_score = float(retrieved.iloc[0]["score"])

    if top_score < MIN_CHUNK_SCORE:
        return {
            "answer": NO_ANSWER_MESSAGE,
            "supported": False,
            "reason": (
                f"Skor sumber terbaik {top_score:.3f} berada di bawah "
                f"ambang {MIN_CHUNK_SCORE:.3f}."
            ),
            "top_score": top_score,
            "sentence_score": None,
            "retrieved": retrieved.to_dict(orient="records"),
            "used_chunk_ids": [],
        }

    candidates = build_sentence_candidates(retrieved)

    ranked_sentences = rank_candidate_sentences(
        candidates=candidates,
        query_vector=query_vector,
        embedder=embedder,
        inference_lock=inference_lock,
    )

    if not ranked_sentences:
        return {
            "answer": NO_ANSWER_MESSAGE,
            "supported": False,
            "reason": "Tidak ditemukan kalimat sumber yang dapat digunakan.",
            "top_score": top_score,
            "sentence_score": None,
            "retrieved": retrieved.to_dict(orient="records"),
            "used_chunk_ids": [],
        }

    best_sentence_score = float(
        ranked_sentences[0]["sentence_score"]
    )

    if best_sentence_score < MIN_SENTENCE_SCORE:
        return {
            "answer": NO_ANSWER_MESSAGE,
            "supported": False,
            "reason": (
                f"Kalimat terbaik memperoleh skor "
                f"{best_sentence_score:.3f}, di bawah ambang "
                f"{MIN_SENTENCE_SCORE:.3f}."
            ),
            "top_score": top_score,
            "sentence_score": best_sentence_score,
            "retrieved": retrieved.to_dict(orient="records"),
            "used_chunk_ids": [],
        }

    answer, source_labels, used_chunk_ids = (
        create_extractive_answer(ranked_sentences)
    )

    return {
        "answer": answer,
        "supported": True,
        "reason": (
            "Jawaban berupa kutipan langsung dari sumber yang lolos "
            "dua tahap pemeriksaan relevansi."
        ),
        "top_score": top_score,
        "sentence_score": best_sentence_score,
        "retrieved": retrieved.to_dict(orient="records"),
        "source_labels": source_labels,
        "used_chunk_ids": used_chunk_ids,
    }


# ============================================================
# 5. KOMPONEN TAMPILAN
# ============================================================
def render_scores(message: dict[str, Any]) -> None:
    top_score = message.get("top_score")
    sentence_score = message.get("sentence_score")

    chips: list[str] = []

    if top_score is not None:
        chips.append(
            f'<span class="score-chip">Chunk {float(top_score):.3f}</span>'
        )

    if sentence_score is not None:
        chips.append(
            '<span class="score-chip">'
            f'Kalimat {float(sentence_score):.3f}'
            "</span>"
        )

    if chips:
        st.markdown(
            '<div class="score-row">'
            + "".join(chips)
            + "</div>",
            unsafe_allow_html=True,
        )


def render_sources(message: dict[str, Any]) -> None:
    retrieved = message.get("retrieved", [])
    if not retrieved:
        return

    supported = bool(message.get("supported"))
    title = (
        "Sumber yang digunakan"
        if supported
        else "Kandidat sumber terdekat"
    )

    used_chunk_ids = set(
        message.get("used_chunk_ids", [])
    )

    with st.expander(title):
        for source in retrieved:
            chunk_id = str(source.get("chunk_id", ""))
            is_used = chunk_id in used_chunk_ids

            badge_text = (
                "Digunakan"
                if is_used
                else "Kandidat"
            )
            badge_class = (
                "source-badge"
                if is_used
                else "source-badge candidate"
            )

            doc_id = html.escape(
                str(source.get("doc_id", "-"))
            )
            safe_chunk_id = html.escape(chunk_id)
            category = html.escape(
                str(source.get("category", "-"))
            )
            source_text = html.escape(
                str(source.get("text", ""))
            )
            score = float(source.get("score", 0.0))

            st.markdown(
                f"""
                <article class="source-card">
                    <div class="source-head">
                        <div class="source-id">
                            {doc_id} · {safe_chunk_id}
                        </div>
                        <span class="{badge_class}">
                            {badge_text}
                        </span>
                    </div>
                    <div class="source-meta">
                        {category} · skor {score:.3f}
                    </div>
                    <div class="source-text">
                        {source_text}
                    </div>
                </article>
                """,
                unsafe_allow_html=True,
            )


def render_assistant_message(message: dict[str, Any]) -> None:
    st.markdown(message["content"])
    render_scores(message)
    render_sources(message)


# ============================================================
# 6. APLIKASI UTAMA
# ============================================================
validate_required_files()

try:
    chunks_df, corpus_embeddings = load_knowledge_base()
except Exception as exc:
    st.error(f"Gagal memuat knowledge base: {exc}")
    st.stop()

st.markdown(
    """
    <section class="rag-hero">
        <div class="rag-brand-row">
            <div class="rag-brand">
                <span class="rag-brand-mark">B</span>
                BBC News Archive
            </div>
            <div class="rag-status">
                <span class="rag-status-dot"></span>
                Siap mencari
            </div>
        </div>

        <h1 class="rag-title">
            Tanya berita,<br><em>bukan tebakan.</em>
        </h1>

        <p class="rag-subtitle">
            Jawaban diambil langsung dari knowledge base berita BBC.
            Ketika sumber tidak cukup relevan, chatbot akan mengatakan
            bahwa pertanyaan tersebut tidak dapat dijawab.
        </p>

        <div class="rag-chips">
            <span class="rag-chip">RAG berbasis sumber</span>
            <span class="rag-chip">Hemat memori</span>
            <span class="rag-chip">Tanpa data real-time</span>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Kontrol aplikasi")
    st.caption(
        "Model hanya mencari jawaban dari berita yang tersimpan "
        "di knowledge base."
    )

    st.metric(
        "Jumlah chunk",
        f"{len(chunks_df):,}".replace(",", "."),
    )
    st.metric(
        "Dimensi embedding",
        int(corpus_embeddings.shape[1]),
    )

    st.caption(
        f"Ambang chunk: {MIN_CHUNK_SCORE:.2f}\n\n"
        f"Ambang kalimat: {MIN_SENTENCE_SCORE:.2f}"
    )

    if st.button(
        "Hapus percakapan",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown(
        """
        <section class="empty-state">
            <div class="empty-title">
                Contoh pertanyaan yang dapat dicoba
            </div>
            <div class="empty-examples">
                <div class="empty-example">
                    <span class="empty-number">1</span>
                    <span>
                        What security threat affected Apple iTunes users?
                    </span>
                </div>
                <div class="empty-example">
                    <span class="empty-number">2</span>
                    <span>
                        What happened to the European software patent proposal?
                    </span>
                </div>
                <div class="empty-example">
                    <span class="empty-number">3</span>
                    <span>
                        Why did China Aviation Oil seek a rescue deal?
                    </span>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

for message in st.session_state.messages:
    role = message["role"]
    avatar = "👤" if role == "user" else "🗞️"

    with st.chat_message(role, avatar=avatar):
        if role == "assistant":
            render_assistant_message(message)
        else:
            st.markdown(message["content"])

question = st.chat_input(
    "Tulis pertanyaan berdasarkan berita BBC..."
)

if question:
    clean_question = normalize_question(question)

    if not clean_question:
        st.warning("Pertanyaan tidak boleh kosong.")
        st.stop()

    user_message = {
        "role": "user",
        "content": clean_question,
    }
    st.session_state.messages.append(user_message)

    with st.chat_message("user", avatar="👤"):
        st.markdown(clean_question)

    with st.chat_message("assistant", avatar="🗞️"):
        with st.spinner(
            "Mencari sumber yang paling relevan..."
        ):
            try:
                embedder, inference_lock = load_embedder()
                result = answer_question(
                    question=clean_question,
                    chunks=chunks_df,
                    corpus_embeddings=corpus_embeddings,
                    embedder=embedder,
                    inference_lock=inference_lock,
                )
            except Exception as exc:
                result = {
                    "answer": (
                        "Aplikasi mengalami kendala ketika memproses "
                        "pertanyaan. Silakan periksa log deployment."
                    ),
                    "supported": False,
                    "reason": str(exc),
                    "top_score": None,
                    "sentence_score": None,
                    "retrieved": [],
                    "used_chunk_ids": [],
                }

        assistant_message = {
            "role": "assistant",
            "content": result["answer"],
            "supported": result.get("supported", False),
            "reason": result.get("reason", ""),
            "top_score": result.get("top_score"),
            "sentence_score": result.get(
                "sentence_score"
            ),
            "retrieved": result.get("retrieved", []),
            "used_chunk_ids": result.get(
                "used_chunk_ids",
                [],
            ),
        }

        render_assistant_message(assistant_message)

    st.session_state.messages.append(
        assistant_message
    )
