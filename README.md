# RAG Customer Service Assistant (Low-Code)

A lightweight Retrieval-Augmented Generation (RAG) chatbot for customer support.
- **No external vector DB**: stores embeddings locally with NumPy.
- **Multi-format ingestion**: `.txt`, `.pdf`, `.docx`.
- **Streamlit UI** with citations to source chunks.
- **OpenAI** for embeddings + responses.

## Quickstart

### 1) Clone / Download
Unzip this folder locally.

### 2) Install
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3) Configure
Copy `.env.example` to `.env` and set your key:
```
OPENAI_API_KEY=sk-...
OPENAI_EMBED_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4o-mini
```

### 4) Add Documents
Put your customer docs in `data/` (FAQs, policies, product manuals). Supported: `.txt`, `.pdf`, `.docx`.

### 5) Run
```bash
streamlit run app/app.py
```
Open the URL Streamlit prints (usually http://localhost:8501).

## How It Works
1. **Ingest**: Parses docs, chunks text, creates embeddings with OpenAI.
2. **Store**: Saves normalized embeddings to `storage/index.npz` and metadata to `storage/meta.jsonl`.
3. **Retrieve**: Finds top-k chunks by cosine similarity.
4. **Generate**: Calls the chat model with retrieved context and returns an answer + citations.

## Notes
- For a small knowledge base, this approach is fast and simple. For bigger ones, swap the retriever for FAISS, Qdrant, or Pinecone.
- You can safely delete `storage/` to re-index from scratch.

## Security
- Do **not** upload confidential data to third-party APIs without proper approval.
- Consider redacting PII or running on self-hosted LLMs if necessary.

## Next Steps (Nice-to-haves)
- Add **MMR**/diversity to retrieval.
- Add **feedback** button to capture “was this helpful?”
- Add **source highlighting** (show exact matched spans).
- Switch to **FastAPI** backend if deploying at scale.
