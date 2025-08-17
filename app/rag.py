import os
import json
import numpy as np
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from openai import OpenAI

def load_index(storage_dir: str = "storage") -> Tuple[np.ndarray, List[Dict]]:
    idx_path = os.path.join(storage_dir, "index.npz")
    meta_path = os.path.join(storage_dir, "meta.jsonl")
    if not os.path.exists(idx_path) or not os.path.exists(meta_path):
        raise RuntimeError("Index not found. Run ingestion first.")
    data = np.load(idx_path)
    vectors = data["vectors"].astype(np.float32)
    meta = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))
    return vectors, meta

def cosine_top_k(query_vec: np.ndarray, mat: np.ndarray, k: int = 5) -> List[int]:
    # both should be normalized
    sims = mat @ query_vec.astype(np.float32)
    # Get top-k indices
    k = min(k, sims.shape[0])
    topk = np.argpartition(-sims, k-1)[:k]
    # sort descending
    topk = topk[np.argsort(-sims[topk])]
    return topk.tolist()

def build_prompt(question: str, contexts: List[Dict]) -> str:
    ctx_blocks = []
    for c in contexts:
        tag = f"{c['source']}#chunk-{c['chunk_id']}"
        ctx_blocks.append(f"[{tag}]\n{c['text']}")
    context_text = "\n\n".join(ctx_blocks[:10])

    prompt = f"""You are a precise, helpful customer support assistant.
Use ONLY the provided context to answer. If the answer isn't in the context, say you don't have that information.
Cite sources inline like [source: filename#chunk-id]. Keep answers concise and polite.

Context:
{context_text}

User question: {question}

Answer with citations: """
    return prompt

def answer(question: str, k: int = 5, storage_dir: str = "storage") -> Dict:
    load_dotenv()
    client = OpenAI()
    embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

    # Load index
    mat, meta = load_index(storage_dir)
    # Embed query
    qvec = client.embeddings.create(model=embed_model, input=[question]).data[0].embedding
    qvec = np.array(qvec, dtype=np.float32)
    qvec = qvec / (np.linalg.norm(qvec) + 1e-10)

    # Retrieve
    idxs = cosine_top_k(qvec, mat, k=k)
    contexts = [meta[i] for i in idxs]

    # Build prompt
    prompt = build_prompt(question, contexts)

    # Generate answer
    resp = client.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": "You are a helpful, accurate customer support assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
    )
    text = resp.choices[0].message.content

    return {
        "answer": text,
        "contexts": contexts
    }
