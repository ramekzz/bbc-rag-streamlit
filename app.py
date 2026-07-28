from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# KONFIGURASI
# ============================================================
APP_DIR = Path(__file__).resolve().parent
CHUNKS_PATH = APP_DIR / "bbc_news_chunks.parquet"

TOP_K = 3
MAX_FEATURES = 30_000
MIN_CHUNK_SCORE = 0.16
MIN_SENTENCE_SCORE = 0.12
MIN_KEYWORD_COVERAGE = 0.30

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

QUESTION_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do",
    "does", "for", "from", "had", "has", "have", "how", "in", "is",
    "it", "of", "on", "or", "that", "the", "this", "to", "was",
    "were", "what", "when", "where", "which", "who", "why", "with",
    "apa", "apakah", "bagaimana", "dalam", "dan", "dari", "di",
    "itu", "ini", "ke", "mengapa", "pada", "siapa", "yang",
}

REQUIRED_COLUMNS = {
    "doc_id",
    "chunk_id",
    "category",
    "chunk_text",
}


# ============================================================
# HALAMAN
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
    --surface: #fffdf9;
    --surface-soft: #f8f3ea;
    --ink: #18211f;
    --muted: #5e6965;
    --line: rgba(24, 33, 31, 0.13);
    --red: #c94131;
    --green: #21483e;
    --green-soft: #e6efe9;
    --shadow: 0 16px 46px rgba(51, 39, 25, 0.09);
}

html {
    color-scheme: light !important;
}

body,
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background:
        radial-gradient(circle at 92% 3%, rgba(201,65,49,.12), transparent 26rem),
        radial-gradient(circle at 4% 96%, rgba(33,72,62,.09), transparent 26rem),
        linear-gradient(180deg, #faf7f1 0%, var(--page) 100%) !important;
    color: var(--ink) !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

#MainMenu,
footer {
    visibility: hidden;
}

.block-container {
    width: min(100%, 900px);
    max-width: 900px;
    padding: 1.35rem 1.15rem 8.2rem;
}

.rag-hero {
    margin-bottom: 1rem;
    padding: 1.55rem;
    border: 1px solid rgba(255,255,255,.8);
    border-radius: 24px;
    background: rgba(255,253,249,.9);
    box-shadow: var(--shadow);
}

.rag-topline {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: .8rem;
    margin-bottom: 1rem;
}

.rag-brand {
    display: flex;
    align-items: center;
    gap: .65rem;
    color: var(--green);
    font-size: .75rem;
    font-weight: 850;
    letter-spacing: .12em;
    text-transform: uppercase;
}

.rag-logo {
    display: grid;
    place-items: center;
    width: 2.15rem;
    height: 2.15rem;
    border-radius: 11px;
    color: white;
    background: var(--red);
    box-shadow: 0 8px 18px rgba(201,65,49,.23);
}

.rag-status {
    display: inline-flex;
    align-items: center;
    gap: .42rem;
    padding: .4rem .65rem;
    border-radius: 999px;
    color: var(--green);
    background: var(--green-soft);
    font-size: .7rem;
    font-weight: 800;
}

.rag-dot {
    width: .48rem;
    height: .48rem;
    border-radius: 50%;
    background: #2e9971;
    box-shadow: 0 0 0 4px rgba(46,153,113,.12);
}

.rag-title {
    margin: 0;
    color: var(--ink);
    font-family: Georgia, "Times New Roman", serif;
    font-size: clamp(2rem, 5vw, 3.3rem);
    line-height: 1;
    letter-spacing: -.045em;
}

.rag-title em {
    color: var(--red);
}

.rag-copy {
    max-width: 700px;
    margin: .9rem 0 0;
    color: var(--muted);
    font-size: .96rem;
    line-height: 1.62;
}

.rag-tags {
    display: flex;
    flex-wrap: wrap;
    gap: .45rem;
    margin-top: 1rem;
}

.rag-tag {
    padding: .4rem .62rem;
    border: 1px solid var(--line);
    border-radius: 999px;
    color: var(--muted);
    background: rgba(255,255,255,.7);
    font-size: .69rem;
    font-weight: 750;
}

.empty-state {
    margin-bottom: 1rem;
    padding: 1rem 1.05rem;
    border: 1px dashed rgba(24,33,31,.2);
    border-radius: 18px;
    color: var(--muted);
    background: rgba(255,253,249,.62);
    font-size: .88rem;
    line-height: 1.55;
}

.empty-state strong {
    color: var(--green);
}

[data-testid="stChatMessage"] {
    min-height: 0 !important;
    margin-bottom: .75rem !important;
    padding: .95rem 1rem !important;
    gap: .75rem !important;
    border: 1px solid var(--line) !important;
    border-radius: 19px !important;
    background: rgba(255,253,249,.96) !important;
    box-shadow: 0 8px 26px rgba(51,39,25,.07) !important;
}

[data-testid="stChatMessage"] > div {
    min-width: 0 !important;
}

[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {
    width: 2.35rem !important;
    height: 2.35rem !important;
    min-width: 2.35rem !important;
    border-radius: 12px !important;
}

[data-testid="stChatMessageContent"],
[data-testid="stChatMessageContent"] *,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] span {
    color: var(--ink) !important;
    -webkit-text-fill-color: var(--ink) !important;
    opacity: 1 !important;
}

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
    font-size: .96rem;
    line-height: 1.6;
    overflow-wrap: anywhere;
}

[data-testid="stChatMessage"] blockquote {
    margin: .7rem 0 !important;
    padding: .75rem .9rem !important;
    border-left: 3px solid var(--red) !important;
    border-radius: 0 11px 11px 0;
    background: var(--surface-soft) !important;
}

[data-testid="stChatMessage"] blockquote p {
    margin: 0 !important;
}

.score-row {
    display: flex;
    flex-wrap: wrap;
    gap: .4rem;
    margin-top: .4rem;
}

.score-chip {
    padding: .3rem .52rem;
    border-radius: 999px;
    color: var(--green);
    background: var(--green-soft);
    font-size: .67rem;
    font-weight: 800;
}

[data-testid="stExpander"] {
    margin-top: .7rem;
    border: 1px solid var(--line) !important;
    border-radius: 14px !important;
    background: rgba(248,243,234,.75) !important;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary * {
    color: var(--ink) !important;
    -webkit-text-fill-color: var(--ink) !important;
}

.source-card {
    margin: .6rem 0;
    padding: .8rem;
    border: 1px solid var(--line);
    border-radius: 13px;
    background: var(--surface);
}

.source-head {
    display: flex;
    justify-content: space-between;
    gap: .75rem;
    margin-bottom: .45rem;
}

.source-id {
    color: var(--green);
    font-size: .78rem;
    font-weight: 850;
    overflow-wrap: anywhere;
}

.source-score {
    flex: 0 0 auto;
    color: var(--red);
    font-size: .7rem;
    font-weight: 850;
}

.source-text {
    color: var(--muted);
    font-size: .8rem;
    line-height: 1.5;
    overflow-wrap: anywhere;
}

[data-testid="stBottom"] {
    padding-bottom: env(safe-area-inset-bottom);
    background:
        linear-gradient(
            180deg,
            rgba(244,239,230,0),
            rgba(244,239,230,.94) 34%,
            rgba(244,239,230,1) 100%
        ) !important;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
}

[data-testid="stBottom"] > div {
    width: min(calc(100% - 2.2rem), 880px) !important;
    margin: 0 auto !important;
    padding-bottom: .75rem !important;
    background: transparent !important;
}

[data-testid="stChatInput"] {
    min-height: 3.55rem !important;
    border: 1px solid rgba(24,33,31,.15) !important;
    border-radius: 18px !important;
    background: rgba(255,253,249,.98) !important;
    box-shadow: 0 15px 38px rgba(51,39,25,.15) !important;
}

[data-testid="stChatInput"] textarea {
    color: var(--ink) !important;
    -webkit-text-fill-color: var(--ink) !important;
    background: transparent !important;
    font-size: .94rem !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #77817d !important;
    -webkit-text-fill-color: #77817d !important;
    opacity: 1 !important;
}

[data-testid="stChatInput"] button {
    color: white !important;
    background: var(--red) !important;
    border-radius: 11px !important;
}

[data-testid="stSidebar"] {
    background: var(--green) !important;
}

[data-testid="stSidebar"] *,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    color: #fffaf1 !important;
    -webkit-text-fill-color: #fffaf1 !important;
}

@media (max-width: 700px) {
    .block-container {
        padding: .65rem .65rem calc(8rem + env(safe-area-inset-bottom));
    }

    .rag-hero {
        padding: 1rem;
        border-radius: 19px;
    }

    .rag-brand {
        font-size: .64rem;
        letter-spacing: .09em;
    }

    .rag-logo {
        width: 1.9rem;
        height: 1.9rem;
        border-radius: 9px;
    }

    .rag-status {
        padding: .32rem .5rem;
        font-size: .61rem;
    }

    .rag-title {
        font-size: clamp(1.8rem, 9vw, 2.35rem);
    }

    .rag-copy {
        font-size: .84rem;
        line-height: 1.5;
    }

    .rag-tags {
        gap: .35rem;
    }

    .rag-tag {
        padding: .34rem .5rem;
        font-size: .61rem;
    }

    [data-testid="stChatMessage"] {
        padding: .75rem .78rem !important;
        gap: .58rem !important;
        border-radius: 16px !important;
    }

    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarAssistant"] {
        width: 2rem !important;
        height: 2rem !important;
        min-width: 2rem !important;
        border-radius: 10px !important;
    }

    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
        font-size: .86rem !important;
        line-height: 1.52 !important;
    }

    [data-testid="stBottom"] > div {
        width: calc(100% - .85rem) !important;
        padding-bottom: calc(.4rem + env(safe-area-inset-bottom)) !important;
    }

    [data-testid="stChatInput"] {
        min-height: 3.25rem !important;
        border-radius: 16px !important;
    }

    [data-testid="stChatInput"] textarea {
        font-size: .86rem !important;
    }
}
</style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA DAN INDEX TF-IDF
# ============================================================
def validate_files() -> None:
    if not CHUNKS_PATH.exists():
        st.error(
            "File bbc_news_chunks.parquet tidak ditemukan. "
            "Letakkan file tersebut di folder yang sama dengan app.py."
        )
        st.stop()


@st.cache_resource(show_spinner="Menyiapkan indeks berita...")
def load_search_index() -> tuple[pd.DataFrame, TfidfVectorizer, Any]:
    chunks = pd.read_parquet(
        CHUNKS_PATH,
        columns=["doc_id", "chunk_id", "category", "chunk_text"],
    ).reset_index(drop=True)

    missing = REQUIRED_COLUMNS.difference(chunks.columns)
    if missing:
        raise ValueError(
            "Kolom parquet belum lengkap: "
            + ", ".join(sorted(missing))
        )

    chunks["chunk_text"] = (
        chunks["chunk_text"]
        .fillna("")
        .astype(str)
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=MAX_FEATURES,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )

    matrix = vectorizer.fit_transform(
        chunks["chunk_text"].tolist()
    )

    return chunks, vectorizer, matrix


# ============================================================
# RETRIEVAL
# ============================================================
def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def asks_for_live_information(question: str) -> bool:
    normalized = normalize_text(question).lower()
    return any(
        marker in normalized
        for marker in LIVE_INFORMATION_MARKERS
    )


def question_keywords(question: str) -> set[str]:
    tokens = re.findall(
        r"[a-zA-Z0-9][a-zA-Z0-9'-]+",
        question.lower(),
    )
    return {
        token
        for token in tokens
        if len(token) > 2
        and token not in QUESTION_STOPWORDS
    }


def keyword_coverage(
    question: str,
    candidate_text: str,
) -> float:
    keywords = question_keywords(question)
    if not keywords:
        return 0.0

    candidate_tokens = set(
        re.findall(
            r"[a-zA-Z0-9][a-zA-Z0-9'-]+",
            candidate_text.lower(),
        )
    )

    matched = keywords.intersection(candidate_tokens)
    return len(matched) / len(keywords)


def retrieve(
    question: str,
    chunks: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    matrix: Any,
) -> tuple[pd.DataFrame, Any]:
    query_vector = vectorizer.transform([question])
    scores = (query_vector @ matrix.T).toarray().ravel()

    if not np.any(scores > 0):
        return pd.DataFrame(), query_vector

    top_k = min(TOP_K, len(scores))
    indices = np.argpartition(scores, -top_k)[-top_k:]
    indices = indices[np.argsort(scores[indices])[::-1]]

    rows: list[dict[str, Any]] = []

    for rank, index in enumerate(indices, start=1):
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


def split_sentences(text: str) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    output: list[str] = []

    for sentence in sentences:
        sentence = sentence.strip()
        words = sentence.split()

        if len(words) < 5:
            continue

        if len(words) <= 75:
            output.append(sentence)
            continue

        for start in range(0, len(words), 55):
            window = words[start:start + 65]
            if len(window) >= 8:
                output.append(" ".join(window))

    return output


def rank_sentences(
    question: str,
    query_vector: Any,
    retrieved: pd.DataFrame,
    vectorizer: TfidfVectorizer,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    for row in retrieved.itertuples():
        for sentence in split_sentences(row.text):
            candidates.append(
                {
                    "sentence": sentence,
                    "doc_id": row.doc_id,
                    "chunk_id": row.chunk_id,
                    "category": row.category,
                    "chunk_score": float(row.score),
                    "coverage": keyword_coverage(
                        question,
                        sentence,
                    ),
                }
            )

    if not candidates:
        return []

    sentence_matrix = vectorizer.transform(
        [item["sentence"] for item in candidates]
    )
    scores = (
        query_vector @ sentence_matrix.T
    ).toarray().ravel()

    ranked: list[dict[str, Any]] = []

    for item, score in zip(candidates, scores):
        result = dict(item)
        result["sentence_score"] = float(score)
        ranked.append(result)

    ranked.sort(
        key=lambda item: (
            item["sentence_score"],
            item["coverage"],
            item["chunk_score"],
        ),
        reverse=True,
    )

    return ranked


def clean_excerpt(text: str, limit: int = 520) -> str:
    text = normalize_text(text)

    if text:
        text = text[0].upper() + text[1:]

    if len(text) <= limit:
        return text

    return text[:limit].rsplit(" ", 1)[0].rstrip(" ,;:-") + "…"


def answer_question(
    question: str,
    chunks: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    matrix: Any,
) -> dict[str, Any]:
    question = normalize_text(question)

    if asks_for_live_information(question):
        return {
            "answer": NO_ANSWER_MESSAGE,
            "supported": False,
            "reason": "Pertanyaan meminta informasi real-time.",
            "top_score": None,
            "sentence_score": None,
            "coverage": None,
            "retrieved": [],
            "used_chunk_ids": [],
        }

    retrieved, query_vector = retrieve(
        question,
        chunks,
        vectorizer,
        matrix,
    )

    if retrieved.empty:
        return {
            "answer": NO_ANSWER_MESSAGE,
            "supported": False,
            "reason": "Tidak ada kecocokan istilah dalam sumber.",
            "top_score": None,
            "sentence_score": None,
            "coverage": None,
            "retrieved": [],
            "used_chunk_ids": [],
        }

    top_score = float(retrieved.iloc[0]["score"])

    if top_score < MIN_CHUNK_SCORE:
        return {
            "answer": NO_ANSWER_MESSAGE,
            "supported": False,
            "reason": "Skor chunk berada di bawah ambang.",
            "top_score": top_score,
            "sentence_score": None,
            "coverage": None,
            "retrieved": retrieved.to_dict(orient="records"),
            "used_chunk_ids": [],
        }

    ranked = rank_sentences(
        question,
        query_vector,
        retrieved,
        vectorizer,
    )

    if not ranked:
        return {
            "answer": NO_ANSWER_MESSAGE,
            "supported": False,
            "reason": "Tidak ditemukan kalimat sumber.",
            "top_score": top_score,
            "sentence_score": None,
            "coverage": None,
            "retrieved": retrieved.to_dict(orient="records"),
            "used_chunk_ids": [],
        }

    best = ranked[0]
    sentence_score = float(best["sentence_score"])
    coverage = float(best["coverage"])

    if (
        sentence_score < MIN_SENTENCE_SCORE
        or coverage < MIN_KEYWORD_COVERAGE
    ):
        return {
            "answer": NO_ANSWER_MESSAGE,
            "supported": False,
            "reason": "Kalimat sumber tidak cukup mendukung pertanyaan.",
            "top_score": top_score,
            "sentence_score": sentence_score,
            "coverage": coverage,
            "retrieved": retrieved.to_dict(orient="records"),
            "used_chunk_ids": [],
        }

    excerpt = clean_excerpt(best["sentence"])
    source = f"{best['doc_id']} · {best['chunk_id']}"

    answer = (
        "Berdasarkan sumber yang tersedia, informasi yang paling "
        "relevan adalah:\n\n"
        f"> {excerpt}\n\n"
        f"**Sumber:** {source}"
    )

    return {
        "answer": answer,
        "supported": True,
        "reason": "Kutipan sumber lolos pemeriksaan relevansi.",
        "top_score": top_score,
        "sentence_score": sentence_score,
        "coverage": coverage,
        "retrieved": retrieved.to_dict(orient="records"),
        "used_chunk_ids": [best["chunk_id"]],
    }


# ============================================================
# KOMPONEN UI
# ============================================================
def render_scores(message: dict[str, Any]) -> None:
    chips: list[str] = []

    if message.get("top_score") is not None:
        chips.append(
            '<span class="score-chip">'
            f'Chunk {float(message["top_score"]):.3f}'
            "</span>"
        )

    if message.get("sentence_score") is not None:
        chips.append(
            '<span class="score-chip">'
            f'Kalimat {float(message["sentence_score"]):.3f}'
            "</span>"
        )

    if message.get("coverage") is not None:
        chips.append(
            '<span class="score-chip">'
            f'Cakupan {float(message["coverage"]) * 100:.0f}%'
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

    used = set(message.get("used_chunk_ids", []))
    title = (
        "Sumber yang digunakan"
        if message.get("supported")
        else "Kandidat sumber terdekat"
    )

    with st.expander(title):
        for source in retrieved:
            doc_id = html.escape(
                str(source.get("doc_id", "-"))
            )
            chunk_id = html.escape(
                str(source.get("chunk_id", "-"))
            )
            source_text = html.escape(
                str(source.get("text", ""))
            )
            score = float(source.get("score", 0.0))
            marker = (
                "digunakan"
                if str(source.get("chunk_id", "")) in used
                else "kandidat"
            )

            st.markdown(
                (
                    '<article class="source-card">'
                    '<div class="source-head">'
                    f'<div class="source-id">{doc_id} · {chunk_id}</div>'
                    f'<div class="source-score">{marker} · {score:.3f}</div>'
                    '</div>'
                    f'<div class="source-text">{source_text}</div>'
                    '</article>'
                ),
                unsafe_allow_html=True,
            )


def render_assistant(message: dict[str, Any]) -> None:
    st.markdown(message["content"])
    render_scores(message)
    render_sources(message)


# ============================================================
# APLIKASI
# ============================================================
validate_files()

try:
    chunks_df, vectorizer, tfidf_matrix = load_search_index()
except Exception as exc:
    st.error(f"Gagal menyiapkan knowledge base: {exc}")
    st.stop()

st.markdown(
    (
        '<section class="rag-hero">'
        '<div class="rag-topline">'
        '<div class="rag-brand">'
        '<span class="rag-logo">B</span>'
        '<span>BBC News Archive</span>'
        '</div>'
        '<div class="rag-status">'
        '<span class="rag-dot"></span>'
        '<span>Siap mencari</span>'
        '</div>'
        '</div>'
        '<h1 class="rag-title">Tanya berita,<br><em>bukan tebakan.</em></h1>'
        '<p class="rag-copy">'
        'Jawaban diambil langsung dari knowledge base berita BBC. '
        'Jika sumber tidak cukup relevan, chatbot akan menolak menjawab.'
        '</p>'
        '<div class="rag-tags">'
        '<span class="rag-tag">RAG berbasis sumber</span>'
        '<span class="rag-tag">Hemat memori</span>'
        '<span class="rag-tag">Tanpa model besar</span>'
        '</div>'
        '</section>'
    ),
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Kontrol aplikasi")
    st.caption(
        "Versi ringan ini menggunakan indeks TF-IDF dan tidak memuat "
        "PyTorch, Transformers, atau Sentence Transformers."
    )
    st.metric(
        "Jumlah chunk",
        f"{len(chunks_df):,}".replace(",", "."),
    )
    st.metric(
        "Jumlah fitur",
        f"{tfidf_matrix.shape[1]:,}".replace(",", "."),
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
        (
            '<div class="empty-state">'
            '<strong>Contoh pertanyaan:</strong><br>'
            'What security threat affected Apple iTunes users?<br>'
            'What happened to the European software patent proposal?<br>'
            'Why did China Aviation Oil seek a rescue deal?'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

for message in st.session_state.messages:
    role = message["role"]
    avatar = "👤" if role == "user" else "🗞️"

    with st.chat_message(role, avatar=avatar):
        if role == "assistant":
            render_assistant(message)
        else:
            st.markdown(message["content"])

question = st.chat_input(
    "Tulis pertanyaan berdasarkan berita BBC..."
)

if question:
    clean_question = normalize_text(question)

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
        with st.spinner("Mencari sumber..."):
            try:
                result = answer_question(
                    clean_question,
                    chunks_df,
                    vectorizer,
                    tfidf_matrix,
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
                    "coverage": None,
                    "retrieved": [],
                    "used_chunk_ids": [],
                }

        assistant_message = {
            "role": "assistant",
            "content": result["answer"],
            "supported": result.get("supported", False),
            "reason": result.get("reason", ""),
            "top_score": result.get("top_score"),
            "sentence_score": result.get("sentence_score"),
            "coverage": result.get("coverage"),
            "retrieved": result.get("retrieved", []),
            "used_chunk_ids": result.get("used_chunk_ids", []),
        }

        render_assistant(assistant_message)

    st.session_state.messages.append(assistant_message)
