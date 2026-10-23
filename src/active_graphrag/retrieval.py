import numpy as np
from .schema import Passage


def chunk_passages(passages, tokenizer, window=500, stride=50):
    if not 0 < stride <= window:
        raise ValueError("Require 0 < stride <= window")
    chunks = []
    for p in passages:
        ids = tokenizer.encode(p.text, add_special_tokens=False)
        if not ids:
            continue
        for start in range(0, len(ids), stride):
            end = min(start + window, len(ids))
            text = p.text if start == 0 and end == len(ids) else tokenizer.decode(ids[start:end], skip_special_tokens=True)
            chunks.append(Passage(f"{p.id}:{start}:{end}", p.title, text, p.id))
            if end == len(ids):
                break
    return chunks


class DenseRetriever:
    def __init__(self, passages, embedder, backend="faiss"):
        self.passages, self.embedder = passages, embedder
        self.vectors = embedder.encode([p.text for p in passages]) if passages else None
        self.index = None
        if backend == "faiss" and passages:
            import faiss
            self.index = faiss.IndexFlatIP(self.vectors.shape[1])
            self.index.add(np.asarray(self.vectors, dtype="float32"))

    def search(self, query, k=3):
        if not self.passages:
            return []
        q = self.embedder.encode([query])
        if self.index is not None:
            # Sort all candidates to make equal-score ordering deterministic.
            values, indices = self.index.search(q, len(self.passages))
            order = sorted(zip(values[0], indices[0]), key=lambda x: (-float(x[0]), int(x[1])))
            selected = [int(i) for _, i in order[:k]]
        else:
            selected = np.argsort(-(self.vectors @ q[0]), kind="stable")[:k]
        return [self.passages[i] for i in selected]
