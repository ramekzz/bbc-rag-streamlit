# BBC News RAG Chatbot dengan Streamlit

Prototipe ini memakai hasil dari notebook `UAS_TRENTOP.ipynb`:

1. `bbc_news_chunks.parquet` sebagai knowledge base.
2. `bbc_chunk_embeddings.npy` sebagai embedding seluruh chunk.
3. `paraphrase-multilingual-MiniLM-L12-v2` untuk retrieval.
4. `google/flan-t5-base` untuk menyusun jawaban.

## Struktur folder

```text
bbc_rag_streamlit/
├── app.py
├── requirements.txt
├── bbc_news_chunks.parquet
└── bbc_chunk_embeddings.npy
```

Dua file data terakhir dibuat oleh notebook dan harus disalin ke folder ini.

## 1. Jalankan notebook sampai bagian penyimpanan

Pastikan sel berikut telah berhasil dijalankan:

```python
chunks_df.to_parquet(
    "bbc_news_chunks.parquet",
    index=False
)

np.save(
    "bbc_chunk_embeddings.npy",
    corpus_embeddings
)
```

## 2. Buat virtual environment

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Instal library

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Salin file hasil notebook

Salin dua file berikut ke folder yang sama dengan `app.py`:

```text
bbc_news_chunks.parquet
bbc_chunk_embeddings.npy
```

## 5. Jalankan aplikasi

```bash
streamlit run app.py
```

Streamlit akan menampilkan alamat lokal, biasanya:

```text
http://localhost:8501
```

## Mekanisme menolak jawaban tanpa sumber

Aplikasi menggunakan dua pagar:

### Pagar 1: ambang retrieval

```python
if top_score < min_retrieval_score:
    return NO_ANSWER_MESSAGE
```

Jika chunk paling mirip masih mempunyai skor di bawah ambang, generator tidak
dipanggil. Nilai awal di aplikasi adalah `0.35`.

### Pagar 2: sentinel dari generator

Prompt memerintahkan generator mengeluarkan `NO_ANSWER` ketika konteks tidak
mengandung jawaban langsung. Aplikasi mengubah sentinel itu menjadi pesan:

```text
Maaf, saya tidak dapat menjawab pertanyaan tersebut karena dokumen yang
tersedia tidak memuat sumber yang cukup relevan.
```

## Menentukan ambang yang lebih baik

Nilai `0.35` hanyalah titik awal, bukan angka universal.

1. Siapkan pertanyaan yang memang dapat dijawab.
2. Siapkan pertanyaan di luar knowledge base.
3. Catat `top_score` masing-masing pertanyaan.
4. Pilih nilai yang memisahkan kedua kelompok dengan kesalahan paling kecil.

Contoh sederhana:

```python
evaluation_questions = [
    ("What security threat affected Apple iTunes users?", 1),
    ("Who won the 2026 football world cup?", 0),
]

records = []

for question, should_be_supported in evaluation_questions:
    retrieved = retrieve_documents(
        question=question,
        top_k=3,
        chunks=chunks_df,
        corpus_embeddings=corpus_embeddings,
        embedder=embedder,
        inference_lock=inference_lock,
    )

    records.append({
        "question": question,
        "should_be_supported": should_be_supported,
        "top_score": float(retrieved.iloc[0]["score"]),
    })

print(pd.DataFrame(records))
```

`1` berarti pertanyaan seharusnya mempunyai sumber, sedangkan `0` berarti
pertanyaan seharusnya ditolak.

## Catatan performa

`google/flan-t5-base` cukup berat pada CPU. Saat pertama kali dijalankan, model
akan diunduh. Proses berikutnya memakai cache lokal. Bila RAM terbatas, Anda
dapat mencoba `google/flan-t5-small`, tetapi kualitas jawabannya biasanya lebih
rendah.
