"""
Retrieval-augmented generation over zoning text.

Documents (PDF/TXT/MD) are split into overlapping chunks, embedded with the
configured LLM provider, and indexed. FAISS (inner product on L2-normalised
vectors) is used when installed; otherwise an equivalent exact NumPy search.
Retrieval for a district combines embedding similarity with a lexical boost for
chunks that contain the district code as a token.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class Document:
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return hashlib.md5(self.content.encode()).hexdigest()[:12]


class DocumentLoader:
    @staticmethod
    def load_file(path: Path) -> List[Document]:
        path = Path(path)
        if path.suffix.lower() == ".pdf":
            from pypdf import PdfReader

            return [Document(p.extract_text() or "", {"source": str(path), "page": i + 1})
                    for i, p in enumerate(PdfReader(path).pages) if (p.extract_text() or "").strip()]
        return [Document(path.read_text(errors="ignore"), {"source": str(path)})]

    @classmethod
    def load_directory(cls, path: Path, extensions=(".pdf", ".txt", ".md")) -> List[Document]:
        docs: List[Document] = []
        for p in sorted(Path(path).rglob("*")):
            if p.suffix.lower() in extensions:
                docs.extend(cls.load_file(p))
        return docs


class TextSplitter:
    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 200):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.size, self.overlap = chunk_size, chunk_overlap

    def split(self, doc: Document) -> List[Document]:
        text = re.sub(r"[ \t]+", " ", doc.content)
        out, start, k = [], 0, 0
        while start < len(text):
            end = min(len(text), start + self.size)
            if end < len(text):
                cut = text.rfind("\n", start + self.size // 2, end)
                end = cut if cut > start else end
            out.append(Document(text[start:end], {**doc.metadata, "chunk": k}))
            k += 1
            if end >= len(text):
                break
            start = max(end - self.overlap, start + 1)
        return out

    def split_documents(self, docs: List[Document]) -> List[Document]:
        return [c for d in docs for c in self.split(d)]


class VectorStore:
    def __init__(self):
        self.docs: List[Document] = []
        self.emb: Optional[np.ndarray] = None
        self._faiss = None

    def add(self, docs: List[Document], embeddings: List[List[float]]):
        e = np.asarray(embeddings, dtype=np.float32)
        e /= np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
        self.docs.extend(docs)
        self.emb = e if self.emb is None else np.vstack([self.emb, e])
        try:
            import faiss

            self._faiss = faiss.IndexFlatIP(self.emb.shape[1])
            self._faiss.add(self.emb)
        except ImportError:
            self._faiss = None

    @property
    def backend(self) -> str:
        return "faiss" if self._faiss is not None else "numpy"

    def search(self, q: List[float], k: int = 5) -> List[tuple]:
        if self.emb is None:
            return []
        q = np.asarray(q, dtype=np.float32)
        q /= max(np.linalg.norm(q), 1e-12)
        k = min(k, len(self.docs))
        if self._faiss is not None:
            s, i = self._faiss.search(q[None], k)
            return [(self.docs[j], float(v)) for j, v in zip(i[0], s[0])]
        sims = self.emb @ q
        top = np.argsort(-sims)[:k]
        return [(self.docs[j], float(sims[j])) for j in top]


class RAGPipeline:
    def __init__(self, llm, splitter: Optional[TextSplitter] = None, cache_path: Optional[Path] = None):
        self.llm = llm
        self.splitter = splitter or TextSplitter()
        self.store = VectorStore()
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: Dict[str, str] = {}
        if self.cache_path and self.cache_path.exists():
            self.cache = json.loads(self.cache_path.read_text())

    def index(self, docs: List[Document]):
        chunks = self.splitter.split_documents(docs)
        self.store.add(chunks, [self.llm.embed(c.content) for c in chunks])
        return len(chunks)

    def retrieve(self, query: str, k: int = 6, must_contain: Optional[str] = None, pool: int = 40):
        hits = self.store.search(self.llm.embed(query), k=pool)
        if must_contain:
            pat = re.compile(rf"(?<![A-Z0-9-]){re.escape(must_contain)}(?![0-9A-Z])")
            hits = sorted(hits, key=lambda h: (not bool(pat.search(h[0].content)), -h[1]))
        return [h[0] for h in hits[:k]]

    def generate(self, query: str, system: str, k: int = 6, must_contain: Optional[str] = None) -> Dict:
        key = hashlib.md5(f"{self.llm.name}|{query}|{must_contain}|{k}".encode()).hexdigest()
        docs = self.retrieve(query, k=k, must_contain=must_contain)
        if key in self.cache:
            answer = self.cache[key]
        else:
            context = "\n\n---\n\n".join(d.content for d in docs)
            prompt = f"Zoning Resolution excerpts:\n\n{context}\n\nTask: {query}"
            answer = self.llm.generate(prompt, system=system)
            self.cache[key] = answer
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                self.cache_path.write_text(json.dumps(self.cache, indent=1))
        return {"answer": answer, "sources": [d.metadata for d in docs]}
