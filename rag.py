# rag.py
from typing import List
import os, glob
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SimpleRAG:
    def __init__(self, docs_dir: str, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.docs_dir = docs_dir
        self.chunks: List[str] = []
        self.emb = None  # 빈 문서일 수 있으므로 None으로 시작
        self._load(docs_dir)

    def _load(self, docs_dir):
        paths = glob.glob(os.path.join(docs_dir, "**/*.txt"), recursive=True)
        if not paths:
            print(f"[RAG] no txt files found under: {docs_dir}")
            self.chunks = []
            self.emb = None
            return

        chunks = []
        for p in paths:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    text = f.read().strip()
            except UnicodeDecodeError:
                with open(p, "r", encoding="cp949", errors="ignore") as f:
                    text = f.read().strip()
            if not text:
                continue
            for chunk in self._chunk(text, 600):
                if chunk.strip():
                    chunks.append(chunk.strip())

        self.chunks = chunks
        if self.chunks:
            self.emb = self.model.encode(self.chunks, convert_to_numpy=True, normalize_embeddings=True)
        else:
            print(f"[RAG] loaded files but chunks are empty: {docs_dir}")
            self.emb = None

    def _chunk(self, text, size):
        # 아주 단순한 라인 기반 청킹
        s = text.splitlines()
        buf, cur = [], 0
        for line in s:
            cur += len(line)
            buf.append(line)
            if cur >= size:
                yield "\n".join(buf)
                buf, cur = [], 0
        if buf:
            yield "\n".join(buf)

    def search(self, query: str, k=3) -> List[str]:
        # 문서가 없거나 임베딩이 없으면 빈 컨텍스트 반환
        if not self.chunks or self.emb is None or len(self.chunks) == 0:
            return []

        q = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)  # (1, d)
        # self.emb shape 확인 (n, d). n==0이면 빈 리스트 반환.
        if self.emb.size == 0 or len(self.emb.shape) != 2:
            return []

        sims = cosine_similarity(q, self.emb)[0]  # (n,)
        if sims.size == 0:
            return []

        idx = sims.argsort()[::-1][:max(1, k)]
        return [self.chunks[i] for i in idx if 0 <= i < len(self.chunks)]
