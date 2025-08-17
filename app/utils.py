import re
from typing import List, Dict

def clean_text(text: str) -> str:
    # Simple normalization
    text = text.replace('\r', '\n')
    text = re.sub('[\t\x00-\x1f]', ' ', text)
    text = re.sub(' +', ' ', text)
    text = re.sub('\n{2,}', '\n\n', text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

def attach_metadata(chunks: List[str], source: str) -> List[Dict]:
    meta = []
    for i, ch in enumerate(chunks):
        meta.append({
            "source": source,
            "chunk_id": i,
            "text": ch
        })
    return meta
