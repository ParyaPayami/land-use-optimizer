"""
LLM provider abstraction (OpenAI, Anthropic, Ollama) plus a test stub.

``MockLLM`` is a deterministic **test stub**: it returns whatever JSON it was
constructed with (default ``{}``) and a hash-based pseudo-embedding. It must not
be used to produce reported extraction results.
"""

from __future__ import annotations

import hashlib
import json
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, system: Optional[str] = None, **kwargs) -> str: ...

    @abstractmethod
    def embed(self, text: str) -> List[float]: ...

    @property
    @abstractmethod
    def name(self) -> str: ...


class OpenAILLM(BaseLLM):
    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None, temperature: float = 0.0,
                 embedding_model: str = "text-embedding-3-small"):
        from openai import OpenAI  # optional dependency

        self.model, self.temperature, self.embedding_model = model, temperature, embedding_model
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    @property
    def name(self) -> str:
        return f"openai/{self.model}"

    def generate(self, prompt: str, system: Optional[str] = None, **kwargs) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        r = self.client.chat.completions.create(model=self.model, messages=msgs,
                                                temperature=kwargs.get("temperature", self.temperature),
                                                max_tokens=kwargs.get("max_tokens", 1000))
        return r.choices[0].message.content

    def embed(self, text: str) -> List[float]:
        return self.client.embeddings.create(model=self.embedding_model, input=text).data[0].embedding


class AnthropicLLM(BaseLLM):
    """Anthropic chat models; embeddings via sentence-transformers (all-MiniLM-L6-v2)."""

    def __init__(self, model: str = "claude-sonnet-4-5", api_key: Optional[str] = None, temperature: float = 0.0):
        from anthropic import Anthropic  # optional dependency

        self.model, self.temperature = model, temperature
        self.client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        self._st = None

    @property
    def name(self) -> str:
        return f"anthropic/{self.model}"

    def generate(self, prompt: str, system: Optional[str] = None, **kwargs) -> str:
        r = self.client.messages.create(model=self.model, max_tokens=kwargs.get("max_tokens", 1000),
                                        temperature=kwargs.get("temperature", self.temperature),
                                        system=system or "You extract numeric zoning rules.",
                                        messages=[{"role": "user", "content": prompt}])
        return r.content[0].text

    def embed(self, text: str) -> List[float]:
        if self._st is None:
            from sentence_transformers import SentenceTransformer

            self._st = SentenceTransformer("all-MiniLM-L6-v2")
        return self._st.encode(text).tolist()


class OllamaLLM(BaseLLM):
    def __init__(self, model: str = "llama3.1", embedding_model: str = "nomic-embed-text", temperature: float = 0.0):
        self.model, self.embedding_model, self.temperature = model, embedding_model, temperature

    @property
    def name(self) -> str:
        return f"ollama/{self.model}"

    def generate(self, prompt: str, system: Optional[str] = None, **kwargs) -> str:
        import ollama

        full = f"{system}\n\n{prompt}" if system else prompt
        return ollama.generate(model=self.model, prompt=full,
                               options={"temperature": kwargs.get("temperature", self.temperature)})["response"]

    def embed(self, text: str) -> List[float]:
        import ollama

        return ollama.embeddings(model=self.embedding_model, prompt=text)["embedding"]


class MockLLM(BaseLLM):
    """Deterministic test stub. Not a model; never use for reported results."""

    def __init__(self, model: str = "stub", response: Optional[Dict] = None):
        self.model = model
        self.response = response or {}

    @property
    def name(self) -> str:
        return f"mock/{self.model}"

    def generate(self, prompt: str, system: Optional[str] = None, **kwargs) -> str:
        return json.dumps(self.response)

    def embed(self, text: str) -> List[float]:
        h = hashlib.sha256(text.encode()).digest()
        return [b / 255.0 for b in h]


def get_llm(provider: str = "mock", model: Optional[str] = None, **kwargs) -> BaseLLM:
    providers = {"openai": OpenAILLM, "anthropic": AnthropicLLM, "ollama": OllamaLLM, "mock": MockLLM}
    if provider not in providers:
        raise ValueError(f"Unknown provider {provider}; choose from {list(providers)}")
    cls = providers[provider]
    return cls(model=model, **kwargs) if model else cls(**kwargs)
