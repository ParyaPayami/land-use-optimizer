# Knowledge Module API Reference

::: pimaluos.knowledge

---

## LLM Abstraction Layer

### BaseLLM

Abstract base class for LLM providers.

::: pimaluos.knowledge.llm.BaseLLM
    options:
      show_source: false
      members:
        - generate
        - embed
        - name

### Provider Implementations

- **OpenAILLM**: GPT-4, GPT-3.5-turbo
- **AnthropicLLM**: Claude-3 models
- **OllamaLLM**: Local models (Llama2, Mistral)
- **MockLLM**: Testing without API keys

```python
from pimaluos.knowledge import get_llm

# Use OpenAI
llm = get_llm("openai", model="gpt-4")

# Use Claude
llm = get_llm("anthropic", model="claude-3-sonnet-20240229")

# Use local model
llm = get_llm("ollama", model="llama2")

# Mock for testing
llm = get_llm("mock")
```

::: pimaluos.knowledge.llm.get_llm

---

## RAG Pipeline

### Document

Document chunk with content and metadata.

::: pimaluos.knowledge.rag.Document

### VectorStore

In-memory vector store with cosine similarity search.

::: pimaluos.knowledge.rag.VectorStore
    options:
      members:
        - add_documents
        - similarity_search
        - save
        - load

### RAGPipeline

Complete RAG pipeline for zoning code queries.

::: pimaluos.knowledge.rag.RAGPipeline
    options:
      members:
        - index_documents
        - retrieve
        - generate

---

## Constraint Parser

### ZoningConstraints

Pydantic model for structured zoning constraints.

::: pimaluos.knowledge.parser.ZoningConstraints

### ConstraintExtractor

Extract structured constraints from zoning regulations.

::: pimaluos.knowledge.parser.ConstraintExtractor
    options:
      members:
        - extract_for_zone
        - validate_proposal

### City-Specific Parsers

- **NYCZoningParser**: NYC special district handling
- **ChicagoZoningParser**: Chicago zoning codes
