"""
PIMALUOS RAG Pipeline

Retrieval-Augmented Generation for zoning code queries:
- Document loading (PDF, text, markdown)
- Vector store with FAISS/Chroma
- Semantic search and retrieval
- Context-aware constraint extraction
"""

from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass
import json
import hashlib
import pickle

import numpy as np


@dataclass
class Document:
    """Document chunk with content and metadata."""
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    
    @property
    def id(self) -> str:
        return hashlib.md5(self.content.encode()).hexdigest()[:12]


class DocumentLoader:
    """Load documents from various sources."""
    
    @staticmethod
    def load_text(path: Path) -> List[Document]:
        """Load plain text file."""
        with open(path) as f:
            content = f.read()
        return [Document(content=content, metadata={"source": str(path), "type": "text"})]
    
    @staticmethod
    def load_pdf(path: Path) -> List[Document]:
        """Load PDF file."""
        try:
            from pypdf import PdfReader
            reader = PdfReader(path)
            docs = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():
                    docs.append(Document(
                        content=text,
                        metadata={"source": str(path), "page": i + 1, "type": "pdf"}
                    ))
            return docs
        except ImportError:
            print("pip install pypdf for PDF support")
            return []
    
    @staticmethod
    def load_markdown(path: Path) -> List[Document]:
        """Load markdown file."""
        with open(path) as f:
            content = f.read()
        return [Document(content=content, metadata={"source": str(path), "type": "markdown"})]
    
    @classmethod
    def load_directory(cls, path: Path, extensions: List[str] = None) -> List[Document]:
        """Load all documents from directory."""
        extensions = extensions or [".txt", ".pdf", ".md"]
        loaders = {
            ".txt": cls.load_text,
            ".pdf": cls.load_pdf,
            ".md": cls.load_markdown,
        }
        
        docs = []
        for ext in extensions:
            for file_path in path.glob(f"**/*{ext}"):
                if ext in loaders:
                    docs.extend(loaders[ext](file_path))
        
        return docs


class TextSplitter:
    """Split documents into chunks for embedding."""
    
    def __init__(
        self, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200,
        separators: List[str] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n## ", "\n### ", "\n\n", "\n", " "]
    
    def split(self, doc: Document) -> List[Document]:
        """Split document into chunks."""
        text = doc.content
        chunks = []
        
        # Recursive splitting
        for sep in self.separators:
            if sep in text:
                parts = text.split(sep)
                current_chunk = ""
                
                for part in parts:
                    if len(current_chunk) + len(part) < self.chunk_size:
                        current_chunk += sep + part if current_chunk else part
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = part
                
                if current_chunk:
                    chunks.append(current_chunk)
                break
        
        if not chunks:
            # Fallback: split by chunk_size
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                chunks.append(text[i:i + self.chunk_size])
        
        return [
            Document(
                content=chunk,
                metadata={**doc.metadata, "chunk_index": i}
            )
            for i, chunk in enumerate(chunks)
        ]
    
    def split_documents(self, docs: List[Document]) -> List[Document]:
        """Split multiple documents."""
        result = []
        for doc in docs:
            result.extend(self.split(doc))
        return result


class VectorStore:
    """
    Vector store for semantic search.
    
    Supports in-memory storage with optional persistence.
    Can be extended to use FAISS or Chroma backends.
    """
    
    def __init__(self, persist_path: Optional[Path] = None):
        self.documents: List[Document] = []
        self.embeddings: np.ndarray = None
        self.persist_path = persist_path
        
        if persist_path and persist_path.exists():
            self.load()
    
    def add_documents(self, docs: List[Document], embeddings: List[List[float]]) -> None:
        """Add documents with their embeddings."""
        for doc, emb in zip(docs, embeddings):
            doc.embedding = emb
            self.documents.append(doc)
        
        self.embeddings = np.array([d.embedding for d in self.documents])
    
    def similarity_search(
        self, 
        query_embedding: List[float], 
        k: int = 5
    ) -> List[Document]:
        """Find k most similar documents."""
        if self.embeddings is None or len(self.documents) == 0:
            return []
        
        query = np.array(query_embedding)
        
        # Cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query)
        similarities = np.dot(self.embeddings, query) / (norms + 1e-10)
        
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        return [self.documents[i] for i in top_indices]
    
    def save(self) -> None:
        """Persist to disk."""
        if self.persist_path:
            data = {
                "documents": [(d.content, d.metadata, d.embedding) for d in self.documents],
            }
            with open(self.persist_path, "wb") as f:
                pickle.dump(data, f)
    
    def load(self) -> None:
        """Load from disk."""
        if self.persist_path and self.persist_path.exists():
            with open(self.persist_path, "rb") as f:
                data = pickle.load(f)
            
            self.documents = [
                Document(content=c, metadata=m, embedding=e)
                for c, m, e in data["documents"]
            ]
            if self.documents:
                self.embeddings = np.array([d.embedding for d in self.documents])


class RAGPipeline:
    """
    Complete RAG pipeline for zoning code queries.
    
    Combines document loading, embedding, retrieval, and LLM generation.
    """
    
    def __init__(
        self, 
        llm,  # BaseLLM instance
        vector_store: VectorStore = None,
        cache_dir: Path = None
    ):
        from pimaluos.knowledge.llm import BaseLLM
        
        self.llm = llm
        self.vector_store = vector_store or VectorStore()
        self.cache_dir = cache_dir or Path("./data/rag_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Response cache
        self.response_cache: Dict[str, str] = {}
        self._load_cache()
    
    def index_documents(self, docs: List[Document]) -> None:
        """Index documents into vector store."""
        print(f"Indexing {len(docs)} documents...")
        
        # Generate embeddings
        embeddings = []
        for i, doc in enumerate(docs):
            emb = self.llm.embed(doc.content)
            embeddings.append(emb)
            if (i + 1) % 10 == 0:
                print(f"  Embedded {i + 1}/{len(docs)}")
        
        self.vector_store.add_documents(docs, embeddings)
        print(f"Indexed {len(docs)} documents")
    
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve relevant documents."""
        query_embedding = self.llm.embed(query)
        return self.vector_store.similarity_search(query_embedding, k=k)
    
    def generate(
        self, 
        query: str, 
        k: int = 5,
        system_prompt: str = None
    ) -> str:
        """Retrieve context and generate response."""
        # Check cache
        cache_key = hashlib.md5(query.encode()).hexdigest()
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]
        
        # Retrieve context
        docs = self.retrieve(query, k=k)
        context = "\n\n---\n\n".join([d.content for d in docs])
        
        # Generate
        prompt = f"""Based on the following zoning regulations:

{context}

Question: {query}

Provide a detailed answer with specific references to the regulations."""
        
        system = system_prompt or "You are a zoning code expert. Extract precise numerical constraints."
        
        response = self.llm.generate(prompt, system=system)
        
        # Cache
        self.response_cache[cache_key] = response
        self._save_cache()
        
        return response
    
    def _load_cache(self) -> None:
        cache_file = self.cache_dir / "response_cache.json"
        if cache_file.exists():
            with open(cache_file) as f:
                self.response_cache = json.load(f)
    
    def _save_cache(self) -> None:
        cache_file = self.cache_dir / "response_cache.json"
        with open(cache_file, "w") as f:
            json.dump(self.response_cache, f)


# Example usage
if __name__ == "__main__":
    from pimaluos.knowledge.llm import get_llm
    
    # Use mock LLM for testing
    llm = get_llm("mock")
    
    # Create pipeline
    pipeline = RAGPipeline(llm)
    
    # Create sample documents
    docs = [
        Document(
            content="R6 districts allow 2.0 FAR and 65 feet height.",
            metadata={"zone": "R6"}
        ),
        Document(
            content="R7 districts allow 3.44 FAR and 85 feet height.",
            metadata={"zone": "R7"}
        ),
    ]
    
    # Index
    pipeline.index_documents(docs)
    
    # Query
    response = pipeline.generate("What is the FAR limit for R6?")
    print(f"Response: {response}")
