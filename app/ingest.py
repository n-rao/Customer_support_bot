import os
import json
import numpy as np
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI

from .utils import clean_text, chunk_text, attach_metadata

try:
    import docx2txt
except Exception:
    docx2txt = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

SUPPORTED_EXTS = {'.txt', '.md', '.pdf', '.docx'}

def read_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.txt', '.md']:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    if ext == '.pdf':
        if PdfReader is None:
            raise RuntimeError("pypdf not installed")
        text = []
        reader = PdfReader(path)
        for page in reader.pages:
            text.append(page.extract_text() or "")
        return "\n".join(text)
    if ext == '.docx':
        if docx2txt is None:
            raise RuntimeError("docx2txt not installed")
        return docx2txt.process(path) or ""
    raise ValueError(f"Unsupported file type: {ext}")

def embed_texts(client: OpenAI, texts: List[str], model: str) -> np.ndarray:
    # Batch to stay within token limits
    vectors = []
    BATCH = 64
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i+BATCH]
        resp = client.embeddings.create(model=model, input=batch)
        batch_vecs = [np.array(d.embedding, dtype=np.float32) for d in resp.data]
        vectors.extend(batch_vecs)
    return np.vstack(vectors) if vectors else np.zeros((0, 1536), dtype=np.float32)

def normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-10
    return mat / norms

def ingest_folder(data_dir: str = "data", storage_dir: str = "storage"):
    load_dotenv()
    client = OpenAI()
    embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

    docs: List[Dict] = []
    for root, _, files in os.walk(data_dir):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext not in SUPPORTED_EXTS:
                continue
            path = os.path.join(root, fn)
            try:
                raw = read_file(path)
                raw = clean_text(raw)
                chunks = chunk_text(raw, chunk_size=1200, overlap=200)
                docs.extend(attach_metadata(chunks, source=os.path.relpath(path, data_dir)))
                print(f"Ingested {fn}: {len(chunks)} chunks")
            except Exception as e:
                print(f"Failed to ingest {path}: {e}")

    texts = [d["text"] for d in docs]
    if not texts:
        raise RuntimeError("No documents found. Put files in the data/ folder.")

    emb = embed_texts(client, texts, model=embed_model)
    emb = normalize_rows(emb)

    os.makedirs(storage_dir, exist_ok=True)
    np.savez_compressed(os.path.join(storage_dir, "index.npz"), vectors=emb)
    with open(os.path.join(storage_dir, "meta.jsonl"), "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"Saved index to {storage_dir}/index.npz and metadata to {storage_dir}/meta.jsonl")
