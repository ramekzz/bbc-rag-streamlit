from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# ============================================================
# 1. KONFIGURASI DASAR
# ============================================================
APP_DIR = Path(__file__).resolve().parent
CHUNKS_PATH = APP_DIR / "bbc_news_chunks.parquet"
EMBEDDINGS_PATH = APP_DIR / "bbc_chunk_embeddings.npy"

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
GENERATION_MODEL_NAME = "google/flan-t5-base"

MAX_INPUT_TOKENS = 512
MAX_CONTEXT_TOKENS = 350

NO_ANSWER_MESSAGE = (
    "Maaf, saya tidak dapat menjawab pertanyaan tersebut karena "
    "dokumen yang tersedia tidak memuat sumber yang cukup relevan."
)

st.set_page_config(
    page_title="BBC News RAG Chatbot",
    page_icon="📰",
    layout="wide",
)

# Tampilan sederhana bertema ruang redaksi.
st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at 85% 5%, rgba(216, 151, 58, .10), transparent 28rem),
                #f6f1e8;
        }
        h1, h2, h3 {
            font-family: Georgia, "Times New Roman", serif;
            letter-spacing: -0.02em;
        }
        [data-testid="stSidebar"] {
            background: #18211f;
        }
        [data-testid="stSidebar"] * {
            color: #f7f0e5;
        }
        [data-testid="stChatMessage"] {
            border: 1px solid rgba(35, 45, 42, .12);
            border-radius: 16px;
            background: rgba(255, 255, 255, .70);
            box-shadow: 0 8px 28px rgba(35, 45, 42, .05);
        }
        .source-note {
            color: #5d665f;
            font-size: .88rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 2. MEMUAT DATA DAN MODEL
# ============================================================
def validate_required_files() -> None:
    """Hentikan aplikasi dengan pesan jelas bila aset belum tersedia."""
    missing = [
        path.name
        for path in (CHUNKS_PATH, EMBEDDINGS_PATH)
        if not path.exists()
    ]

    if missing:
        st.error(
            "File hasil notebook belum ditemukan: "
            + ", ".join(missing)
            + ". Jalankan notebook sampai bagian penyimpanan model, "
              "kemudian salin file tersebut ke folder yang sama dengan app.py."
        )
        st.stop()


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, np.ndarray]:
    """Data tabular dan embedding disimpan di cache data Streamlit."""
    chunks = pd.read_parquet(CHUNKS_PATH)
    embeddings = np.load(EMBEDDINGS_PATH, allow_pickle=False)

    required_columns = {
        "doc_id",
        "chunk_id",
        "category",
        "chunk_text",
    }
    missing_columns = required_columns.difference(chunks.columns)

    if missing_columns:
        raise ValueError(
            "Kolom pada bbc_news_chunks.parquet tidak lengkap: "
            + ", ".join(sorted(missing_columns))
        )

    if len(chunks) != len(embeddings):
        raise ValueError(
            "Jumlah baris chunk tidak sama dengan jumlah embedding. "
            f"Chunk={len(chunks)}, embedding={len(embeddings)}."
        )

    return chunks.reset_index(drop=True), embeddings


@st.cache_resource(show_spinner="Memuat model AI...")
def load_models() -> tuple[
    SentenceTransformer,
    Any,
    AutoModelForSeq2SeqLM,
    str,
    threading.Lock,
]:
    """
    Model adalah resource besar, sehingga dimuat satu kali dan dipakai ulang.
    Lock mencegah beberapa proses inferensi memakai model pada saat bersamaan.
    """
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL_NAME)
    generator = AutoModelForSeq2SeqLM.from_pretrained(
        GENERATION_MODEL_NAME
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = generator.to(device)
    generator.eval()

    inference_lock = threading.Lock()
    return embedder, tokenizer, generator, device, inference_lock


# ============================================================
# 3. RETRIEVAL: MENCARI CHUNK PALING RELEVAN
# ============================================================
def retrieve_documents(
    question: str,
    top_k: int,
    chunks: pd.DataFrame,
    corpus_embeddings: np.ndarray,
    embedder: SentenceTransformer,
    inference_lock: threading.Lock,
) -> pd.DataFrame:
    """Cari top-k chunk menggunakan kemiripan embedding."""
    with inference_lock:
        if hasattr(embedder, "encode_query"):
            query_embedding = embedder.encode_query(
                [question],
                normalize_embeddings=True,
            )
        else:
            query_embedding = embedder.encode(
                [question],
                normalize_embeddings=True,
            )

    hits = util.semantic_search(
        query_embedding,
        corpus_embeddings,
        top_k=min(top_k, len(chunks)),
    )[0]

    rows: list[dict[str, Any]] = []

    for rank, hit in enumerate(hits, start=1):
        row = chunks.iloc[hit["corpus_id"]]
        rows.append(
            {
                "rank": rank,
                "doc_id": row["doc_id"],
                "chunk_id": row["chunk_id"],
                "category": row["category"],
                "score": float(hit["score"]),
                "text": row["chunk_text"],
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 4. MEMBATASI KONTEKS AGAR TIDAK TERPOTONG SEMBARANGAN
# ============================================================
def build_context(
    retrieved: pd.DataFrame,
    tokenizer: Any,
    max_context_tokens: int = MAX_CONTEXT_TOKENS,
) -> tuple[str, list[str]]:
    """Susun konteks tanpa melewati anggaran token FLAN-T5."""
    context_parts: list[str] = []
    used_doc_ids: list[str] = []
    remaining_tokens = max_context_tokens

    for row in retrieved.itertuples():
        part = (
            f"[SOURCE doc_id={row.doc_id}; "
            f"chunk_id={row.chunk_id}; category={row.category}]\n"
            f"{row.text}"
        )

        token_ids = tokenizer.encode(
            part,
            add_special_tokens=False,
        )

        if len(token_ids) <= remaining_tokens:
            context_parts.append(part)
            used_doc_ids.append(str(row.doc_id))
            remaining_tokens -= len(token_ids)
            continue

        # Pakai sisa ruang untuk sebagian chunk, bila masih cukup berarti.
        if remaining_tokens >= 40:
            shortened = tokenizer.decode(
                token_ids[:remaining_tokens],
                skip_special_tokens=True,
            )
            context_parts.append(shortened)
            used_doc_ids.append(str(row.doc_id))

        break

    return "\n\n".join(context_parts), used_doc_ids


# ============================================================
# 5. GENERATOR DAN MEKANISME "TIDAK TAHU"
# ============================================================
def model_indicates_no_answer(answer: str) -> bool:
    """Kenali sentinel atau frasa penolakan yang mungkin dihasilkan model."""
    normalized = re.sub(r"\s+", " ", answer).strip().lower()

    refusal_markers = (
        "no_answer",
        "not enough information",
        "does not provide enough information",
        "do not provide enough information",
        "cannot be answered from the context",
        "tidak cukup informasi",
        "tidak dapat dijawab dari konteks",
    )
    return not normalized or any(
        marker in normalized for marker in refusal_markers
    )


def generate_grounded_answer(
    question: str,
    retrieved: pd.DataFrame,
    min_retrieval_score: float,
    tokenizer: Any,
    generator: AutoModelForSeq2SeqLM,
    device: str,
    inference_lock: threading.Lock,
) -> dict[str, Any]:
    """
    Dua pagar penolakan:
    1. Tolak sebelum generasi jika skor sumber teratas di bawah ambang.
    2. Instruksikan model mengeluarkan NO_ANSWER bila konteks tidak menjawab.
    """
    if retrieved.empty:
        return {
            "answer": NO_ANSWER_MESSAGE,
            "supported": False,
            "used_doc_ids": [],
            "top_score": None,
            "reason": "Tidak ada dokumen yang berhasil diambil.",
        }

    top_score = float(retrieved.iloc[0]["score"])

    # Pagar pertama: tidak ada sumber yang cukup dekat dengan pertanyaan.
    if top_score < min_retrieval_score:
        return {
            "answer": NO_ANSWER_MESSAGE,
            "supported": False,
            "used_doc_ids": [],
            "top_score": top_score,
            "reason": (
                f"Skor sumber terbaik {top_score:.3f} berada di bawah "
                f"ambang {min_retrieval_score:.3f}."
            ),
        }

    context, used_doc_ids = build_context(retrieved, tokenizer)

    prompt = f"""
You are a grounded question-answering assistant.

QUESTION:
{question}

INSTRUCTIONS:
- Use only facts explicitly stated in CONTEXT.
- Never use outside knowledge and never guess.
- If CONTEXT does not directly answer QUESTION, output exactly: NO_ANSWER
- Answer in the same language as QUESTION.
- Keep the answer concise.
- End a supported answer with the relevant document IDs in square brackets.

CONTEXT:
{context}

ANSWER:
""".strip()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=MAX_INPUT_TOKENS,
        truncation=True,
    ).to(device)

    with inference_lock, torch.inference_mode():
        output_ids = generator.generate(
            **inputs,
            max_new_tokens=140,
            do_sample=False,
            num_beams=4,
            early_stopping=True,
        )

    raw_answer = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
    ).strip()

    # Pagar kedua: model menilai konteks tidak cukup.
    if model_indicates_no_answer(raw_answer):
        return {
            "answer": NO_ANSWER_MESSAGE,
            "supported": False,
            "used_doc_ids": [],
            "top_score": top_score,
            "reason": "Generator menilai konteks tidak memuat jawaban langsung.",
        }

    return {
        "answer": raw_answer,
        "supported": True,
        "used_doc_ids": used_doc_ids,
        "top_score": top_score,
        "reason": "Jawaban dibuat dari konteks yang lolos ambang retrieval.",
    }


def answer_question(
    question: str,
    top_k: int,
    min_retrieval_score: float,
    chunks: pd.DataFrame,
    corpus_embeddings: np.ndarray,
    embedder: SentenceTransformer,
    tokenizer: Any,
    generator: AutoModelForSeq2SeqLM,
    device: str,
    inference_lock: threading.Lock,
) -> dict[str, Any]:
    """Pipeline lengkap: pertanyaan -> retrieval -> pemeriksaan -> jawaban."""
    retrieved = retrieve_documents(
        question=question,
        top_k=top_k,
        chunks=chunks,
        corpus_embeddings=corpus_embeddings,
        embedder=embedder,
        inference_lock=inference_lock,
    )

    result = generate_grounded_answer(
        question=question,
        retrieved=retrieved,
        min_retrieval_score=min_retrieval_score,
        tokenizer=tokenizer,
        generator=generator,
        device=device,
        inference_lock=inference_lock,
    )
    result["retrieved"] = retrieved.to_dict(orient="records")
    return result


# ============================================================
# 6. ANTARMUKA STREAMLIT
# ============================================================
validate_required_files()

try:
    chunks_df, corpus_embeddings = load_data()
    (
        embedder,
        tokenizer,
        generator,
        device,
        inference_lock,
    ) = load_models()
except Exception as exc:
    st.exception(exc)
    st.stop()

with st.sidebar:
    st.markdown("## Pengaturan")
    top_k = st.slider(
        "Jumlah kandidat sumber",
        min_value=1,
        max_value=5,
        value=3,
        help="Berapa banyak chunk terdekat yang diperiksa.",
    )
    min_retrieval_score = st.slider(
        "Ambang relevansi minimum",
        min_value=0.00,
        max_value=1.00,
        value=0.35,
        step=0.01,
        help=(
            "Jika skor sumber terbaik lebih rendah dari nilai ini, "
            "chatbot menolak menjawab."
        ),
    )
    show_debug = st.toggle(
        "Tampilkan detail retrieval",
        value=True,
    )

    st.divider()
    st.caption(f"Perangkat inferensi: **{device.upper()}**")
    st.caption(f"Jumlah chunk: **{len(chunks_df):,}**")

    if st.button("Hapus percakapan", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("BBC News · Source-Grounded Chat")
st.caption(
    "Chatbot hanya menjawab berdasarkan chunk berita yang tersimpan. "
    "Saat sumber tidak cukup relevan, chatbot akan menolak menjawab."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            "Silakan ajukan pertanyaan tentang berita dalam knowledge base. "
            "Contoh: **What security threat affected Apple iTunes users?**"
        )

# Tampilkan ulang riwayat percakapan pada setiap rerun.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            top_score = message.get("top_score")
            if top_score is not None:
                st.caption(
                    f"Skor sumber terbaik: {top_score:.3f} · "
                    f"Ambang: {message['threshold']:.3f}"
                )

            retrieved_records = message.get("retrieved", [])

            if message.get("supported") and retrieved_records:
                with st.expander("Lihat sumber jawaban"):
                    source_df = pd.DataFrame(retrieved_records)
                    st.dataframe(
                        source_df[
                            [
                                "rank",
                                "doc_id",
                                "chunk_id",
                                "category",
                                "score",
                            ]
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )

                    for source in retrieved_records:
                        st.markdown(
                            f"**{source['doc_id']} · "
                            f"{source['chunk_id']} · "
                            f"score {source['score']:.3f}**"
                        )
                        st.write(source["text"])

            elif show_debug and retrieved_records:
                with st.expander(
                    "Kandidat terdekat — tidak dipakai sebagai sumber"
                ):
                    candidate_df = pd.DataFrame(retrieved_records)
                    st.dataframe(
                        candidate_df[
                            [
                                "rank",
                                "doc_id",
                                "chunk_id",
                                "category",
                                "score",
                            ]
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.caption(message.get("reason", ""))

question = st.chat_input(
    "Tulis pertanyaan berdasarkan berita BBC...",
    max_chars=500,
)

if question:
    clean_question = question.strip()

    if clean_question:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": clean_question,
            }
        )

        with st.chat_message("user"):
            st.markdown(clean_question)

        with st.chat_message("assistant"):
            with st.spinner("Mencari sumber dan menyusun jawaban..."):
                try:
                    result = answer_question(
                        question=clean_question,
                        top_k=top_k,
                        min_retrieval_score=min_retrieval_score,
                        chunks=chunks_df,
                        corpus_embeddings=corpus_embeddings,
                        embedder=embedder,
                        tokenizer=tokenizer,
                        generator=generator,
                        device=device,
                        inference_lock=inference_lock,
                    )
                except Exception as exc:
                    st.error("Terjadi kesalahan saat memproses pertanyaan.")
                    st.exception(exc)
                    st.stop()

            st.markdown(result["answer"])

            if result["top_score"] is not None:
                st.caption(
                    f"Skor sumber terbaik: {result['top_score']:.3f} · "
                    f"Ambang: {min_retrieval_score:.3f}"
                )

            retrieved_records = result["retrieved"]

            if result["supported"] and retrieved_records:
                with st.expander("Lihat sumber jawaban"):
                    source_df = pd.DataFrame(retrieved_records)
                    st.dataframe(
                        source_df[
                            [
                                "rank",
                                "doc_id",
                                "chunk_id",
                                "category",
                                "score",
                            ]
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )

                    for source in retrieved_records:
                        st.markdown(
                            f"**{source['doc_id']} · "
                            f"{source['chunk_id']} · "
                            f"score {source['score']:.3f}**"
                        )
                        st.write(source["text"])

            elif show_debug and retrieved_records:
                with st.expander(
                    "Kandidat terdekat — tidak dipakai sebagai sumber"
                ):
                    candidate_df = pd.DataFrame(retrieved_records)
                    st.dataframe(
                        candidate_df[
                            [
                                "rank",
                                "doc_id",
                                "chunk_id",
                                "category",
                                "score",
                            ]
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.caption(result["reason"])

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "supported": result["supported"],
                "top_score": result["top_score"],
                "threshold": min_retrieval_score,
                "reason": result["reason"],
                "retrieved": retrieved_records,
            }
        )
