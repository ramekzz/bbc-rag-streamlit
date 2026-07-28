from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Any

# Membatasi thread agar lebih ramah terhadap resource Streamlit Community Cloud.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. KONFIGURASI
# ============================================================
APP_DIR = Path(__file__).resolve().parent
CHUNKS_PATH = APP_DIR / "bbc_news_chunks.parquet"
EMBEDDINGS_PATH = APP_DIR / "bbc_chunk_embeddings.npy"

# Harus sama dengan model yang digunakan ketika membuat embedding korpus.
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
# 2. KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="BBC News RAG Chatbot",
    page_icon="📰",
    layout="wide",
)

st.markdown(
    """
    <style>
        :root {
            --ink: #19211f;
            --paper: #f5f0e7;
            --card: rgba(255, 255, 255, 0.82);
            --accent: #c14924;
            --muted: #68716d;
            --line: rgba(25, 33, 31, 0.13);
        }

        .stApp {
            background:
                radial-gradient(circle at 88% 5%,
                    rgba(193, 73, 36, 0.10),
                    transparent 30rem),
                linear-gradient(180deg, #f8f4ed 0%, var(--paper) 100%);
            color: var(--ink);
        }

        .block-container {
            max-width: 980px;
            padding-top: 2.5rem;
            padding-bottom: 5rem;
        }

        h1, h2, h3 {
            font-family: Georgia, "Times New Roman", serif;
            letter-spacing: -0.025em;
        }

        [data-testid="stSidebar"] {
            background: #17201e;
        }

        [data-testid="stSidebar"] * {
            color: #f8f1e6;
        }

        [data-testid="stChatMessage"] {
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: 18px;
            box-shadow: 0 10px 30px rgba(25, 33, 31, 0.05);
            padding: 0.25rem 0.35rem;
        }

        [data-testid="stChatInput"] {
            border-color: var(--line);
        }

        .eyebrow {
            color: var(--accent);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }

        .intro {
            color: var(--muted);
            font-size: 1rem;
            line-height: 1.65;
            max-width: 760px;
            margin-bottom: 1.2rem;
        }

        .status-note {
            color: var(--muted);
            font-size: 0.88rem;
        }

        div[data-testid="stExpander"] {
            background: rgba(255, 255, 255, 0.58);
            border: 1px solid var(--line);
            border-radius: 14px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. VALIDASI DAN PEMUATAN DATA
# ============================================================
def validate_required_files() -> None:
    """Menghentikan aplikasi jika knowledge base belum tersedia."""
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
    """
    Memuat data satu kali untuk seluruh sesi.

    Embedding dibaca dengan memory mapping agar seluruh matriks tidak
    langsung disalin ke RAM.
    """
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
        raise ValueError(
            "Embedding harus berupa matriks dua dimensi."
        )

    if len(chunks) != embeddings.shape[0]:
        raise ValueError(
            "Jumlah chunk dan embedding berbeda. "
            f"Chunk={len(chunks)}, embedding={embeddings.shape[0]}."
        )

    return chunks, embeddings


@st.cache_resource(show_spinner="Memuat model pencarian...")
def load_embedder() -> tuple[SentenceTransformer, threading.Lock]:
    """
    Hanya memuat satu model embedding.

    Versi ini sengaja tidak memuat FLAN-T5 agar penggunaan RAM tetap
    berada di bawah batas Streamlit Community Cloud.
    """
    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME,
        device="cpu",
    )
    model.eval()
    return model, threading.Lock()


# ============================================================
# 4. FUNGSI BANTU
# ============================================================
def normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", question).strip()


def asks_for_live_information(question: str) -> bool:
    """Knowledge base bersifat statis, bukan sumber data real-time."""
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
    """Membuat embedding pertanyaan yang sudah dinormalisasi."""
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
    """
    Menghitung cosine similarity per blok.

    Cara ini menghindari penyalinan seluruh embedding ke tensor PyTorch.
    """
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
        denominators = np.maximum(norms, 1e-12)

        scores[start:end] = (
            block @ normalized_query
        ) / denominators

    return scores


def retrieve_documents(
    question: str,
    chunks: pd.DataFrame,
    corpus_embeddings: np.ndarray,
    embedder: SentenceTransformer,
    inference_lock: threading.Lock,
    top_k: int = TOP_K,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Mengambil chunk dengan skor cosine tertinggi."""
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
    """
    Memecah chunk menjadi kalimat.

    Kalimat yang terlalu panjang dipecah lagi menjadi jendela kata agar
    proses ranking tetap ringan dan jawaban tidak terlalu panjang.
    """
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
    """Membangun kandidat kalimat hanya dari chunk hasil retrieval."""
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
    """Merangking kalimat kandidat dengan model embedding yang sama."""
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
    """Menjaga kutipan tetap ringkas tanpa mengubah isi sumber."""
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) <= max_characters:
        return text

    shortened = text[:max_characters].rsplit(" ", 1)[0]
    return shortened.rstrip(" ,;:-") + "…"


def create_extractive_answer(
    ranked_sentences: list[dict[str, Any]],
) -> tuple[str, list[str], list[str]]:
    """
    Menampilkan kutipan sumber secara langsung.

    Tidak ada model generatif, sehingga aplikasi tidak menambahkan fakta
    yang tidak terdapat pada knowledge base.
    """
    best = ranked_sentences[0]

    selected = [best]

    # Tambahkan satu kalimat pendukung hanya jika sangat dekat dengan hasil
    # terbaik dan tidak identik.
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

    source_labels = []
    used_chunk_ids = []

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
        "Berdasarkan sumber yang tersedia, informasi yang paling relevan adalah:\n\n"
        f"{quoted_text}\n\n"
        f"**Sumber:** {', '.join(source_labels)}"
    )

    return answer, source_labels, used_chunk_ids


# ============================================================
# 5. PIPELINE JAWABAN
# ============================================================
def answer_question(
    question: str,
    chunks: pd.DataFrame,
    corpus_embeddings: np.ndarray,
    embedder: SentenceTransformer,
    inference_lock: threading.Lock,
) -> dict[str, Any]:
    """Pertanyaan -> retrieval -> validasi -> kutipan sumber."""
    question = normalize_question(question)

    # Pertanyaan real-time ditolak sebelum semantic search.
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

    # Pagar pertama: skor chunk.
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

    # Pagar kedua: skor kalimat.
    if best_sentence_score < MIN_SENTENCE_SCORE:
        return {
            "answer": NO_ANSWER_MESSAGE,
            "supported": False,
            "reason": (
                f"Kalimat terbaik hanya memperoleh skor "
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
# 6. TAMPILAN SUMBER
# ============================================================
def render_sources(message: dict[str, Any]) -> None:
    retrieved = message.get("retrieved", [])
    if not retrieved:
        return

    supported = bool(message.get("supported"))

    title = (
        "Lihat sumber jawaban"
        if supported
        else "Lihat kandidat terdekat"
    )

    with st.expander(title):
        source_df = pd.DataFrame(retrieved)

        visible_columns = [
            column
            for column in (
                "rank",
                "doc_id",
                "chunk_id",
                "category",
                "score",
            )
            if column in source_df.columns
        ]

        if visible_columns:
            st.dataframe(
                source_df[visible_columns],
                use_container_width=True,
                hide_index=True,
            )

        used_chunk_ids = set(
            message.get("used_chunk_ids", [])
        )

        for source in retrieved:
            chunk_id = str(source.get("chunk_id", ""))
            is_used = chunk_id in used_chunk_ids

            label = (
                "Dipakai sebagai sumber"
                if is_used
                else "Kandidat retrieval"
            )

            st.markdown(
                f"**{source.get('doc_id', '-')} · "
                f"{chunk_id} · "
                f"skor {float(source.get('score', 0.0)):.3f}**  \n"
                f"<span class='status-note'>{label}</span>",
                unsafe_allow_html=True,
            )
            st.write(source.get("text", ""))


# ============================================================
# 7. ANTARMUKA UTAMA
# ============================================================
validate_required_files()

try:
    chunks_df, corpus_embeddings = load_knowledge_base()
except Exception as exc:
    st.error(f"Gagal memuat knowledge base: {exc}")
    st.stop()

st.markdown(
    '<div class="eyebrow">Source-grounded assistant</div>',
    unsafe_allow_html=True,
)
st.title("BBC News RAG Chatbot")
st.markdown(
    """
    <div class="intro">
        Chatbot ini mencari jawaban hanya dari knowledge base berita BBC.
        Jawaban ditampilkan sebagai kutipan sumber, bukan hasil tebakan model.
        Pertanyaan yang tidak memiliki sumber relevan akan ditolak.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Tentang aplikasi")
    st.caption(
        "Prototype UAS Trending Topics on Statistics "
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
        f"Ambang chunk: {MIN_CHUNK_SCORE:.2f} · "
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

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            top_score = message.get("top_score")
            sentence_score = message.get("sentence_score")

            score_parts = []

            if top_score is not None:
                score_parts.append(
                    f"skor chunk {top_score:.3f}"
                )

            if sentence_score is not None:
                score_parts.append(
                    f"skor kalimat {sentence_score:.3f}"
                )

            if score_parts:
                st.caption(" · ".join(score_parts))

            render_sources(message)

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

    with st.chat_message("user"):
        st.markdown(clean_question)

    with st.chat_message("assistant"):
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

        st.markdown(result["answer"])

        top_score = result.get("top_score")
        sentence_score = result.get("sentence_score")

        score_parts = []

        if top_score is not None:
            score_parts.append(
                f"skor chunk {top_score:.3f}"
            )

        if sentence_score is not None:
            score_parts.append(
                f"skor kalimat {sentence_score:.3f}"
            )

        if score_parts:
            st.caption(" · ".join(score_parts))

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

        render_sources(assistant_message)

    st.session_state.messages.append(
        assistant_message
    )
