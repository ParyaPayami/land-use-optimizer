"""
PIMALUOS Knowledge Module

LLM-RAG system for zoning constraint extraction:
- Multi-LLM abstraction layer (OpenAI, Claude, Ollama)
- RAG pipeline with vector search
- Structured constraint extraction with Pydantic
"""

from .llm import (
    BaseLLM,
    OpenAILLM,
    AnthropicLLM,
    OllamaLLM,
    MockLLM,
    get_llm,
)
from .rag import (
    Document,
    DocumentLoader,
    TextSplitter,
    VectorStore,
    RAGPipeline,
)
from .parser import (
    ZoningConstraints,
    UseRegulations,
    BulkRegulations,
    LotRequirements,
    YardRequirements,
    ParkingRequirements,
    InclusionaryHousing,
    ConstraintCache,
    ConstraintExtractor,
    NYCZoningParser,
    ChicagoZoningParser,
)

__all__ = [
    # LLM
    "BaseLLM",
    "OpenAILLM",
    "AnthropicLLM",
    "OllamaLLM",
    "MockLLM",
    "get_llm",
    
    # RAG
    "Document",
    "DocumentLoader",
    "TextSplitter",
    "VectorStore",
    "RAGPipeline",
    
    # Parser
    "ZoningConstraints",
    "UseRegulations",
    "BulkRegulations",
    "LotRequirements",
    "YardRequirements",
    "ParkingRequirements",
    "InclusionaryHousing",
    "ConstraintCache",
    "ConstraintExtractor",
    "NYCZoningParser",
    "ChicagoZoningParser",
]
