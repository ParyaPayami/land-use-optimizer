"""
PIMALUOS LLM Abstraction Layer

Provides unified interface for multiple LLM providers:
- OpenAI (GPT-4, GPT-3.5-turbo)
- Anthropic (Claude-3)
- Ollama (local models like Llama2, Mistral)
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from pathlib import Path
import os


class BaseLLM(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Generate embedding vector for text."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return model name."""
        pass


class OpenAILLM(BaseLLM):
    """OpenAI GPT models."""
    
    def __init__(
        self, 
        model: str = "gpt-4", 
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        embedding_model: str = "text-embedding-3-small"
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.temperature = temperature
        self.embedding_model = embedding_model
        self._client = None
    
    @property
    def client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("pip install openai")
        return self._client
    
    @property
    def name(self) -> str:
        return f"openai/{self.model}"
    
    def generate(self, prompt: str, system: str = None, **kwargs) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", 2000),
        )
        return response.choices[0].message.content
    
    def embed(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding


class AnthropicLLM(BaseLLM):
    """Anthropic Claude models."""
    
    def __init__(
        self, 
        model: str = "claude-3-sonnet-20240229", 
        api_key: Optional[str] = None,
        temperature: float = 0.0
    ):
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.temperature = temperature
        self._client = None
    
    @property
    def client(self):
        if self._client is None:
            try:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("pip install anthropic")
        return self._client
    
    @property
    def name(self) -> str:
        return f"anthropic/{self.model}"
    
    def generate(self, prompt: str, system: str = None, **kwargs) -> str:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", 2000),
            system=system or "You are a zoning and urban planning expert.",
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    
    def embed(self, text: str) -> List[float]:
        # Claude doesn't have native embeddings, use sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            return model.encode(text).tolist()
        except ImportError:
            raise ImportError("pip install sentence-transformers")


class OllamaLLM(BaseLLM):
    """Ollama local models."""
    
    def __init__(
        self, 
        model: str = "llama2", 
        host: str = "http://localhost:11434",
        temperature: float = 0.0
    ):
        self.model = model
        self.host = host
        self.temperature = temperature
    
    @property
    def name(self) -> str:
        return f"ollama/{self.model}"
    
    def generate(self, prompt: str, system: str = None, **kwargs) -> str:
        try:
            import ollama
            
            full_prompt = f"{system}\n\n{prompt}" if system else prompt
            response = ollama.generate(
                model=self.model,
                prompt=full_prompt,
                options={"temperature": kwargs.get("temperature", self.temperature)}
            )
            return response["response"]
        except ImportError:
            raise ImportError("pip install ollama")
    
    def embed(self, text: str) -> List[float]:
        try:
            import ollama
            response = ollama.embeddings(model=self.model, prompt=text)
            return response["embedding"]
        except ImportError:
            raise ImportError("pip install ollama")


class MockLLM(BaseLLM):
    """Mock LLM for testing without API keys."""
    
    def __init__(self, model: str = "mock"):
        self.model = model
    
    @property
    def name(self) -> str:
        return f"mock/{self.model}"
    
    def generate(self, prompt: str, **kwargs) -> str:
        # Return structured JSON for constraint extraction
        if "constraints" in prompt.lower() or "zoning" in prompt.lower():
            return '''
{
    "permitted_uses": ["residential", "community_facility"],
    "prohibited_uses": ["industrial", "heavy_manufacturing"],
    "max_far": 2.0,
    "max_height_ft": 65,
    "max_stories": 6,
    "min_lot_area_sqft": 1700,
    "min_lot_width_ft": 18,
    "front_yard_ft": 10,
    "side_yard_ft": 8,
    "rear_yard_ft": 30,
    "parking_ratio": 0.5,
    "special_conditions": []
}'''
        return "Mock response for: " + prompt[:50]
    
    def embed(self, text: str) -> List[float]:
        import hashlib
        # Generate deterministic pseudo-embedding from text hash
        hash_bytes = hashlib.sha256(text.encode()).digest()
        return [float(b) / 255.0 for b in hash_bytes[:384]]


def get_llm(provider: str = "openai", model: Optional[str] = None, **kwargs) -> BaseLLM:
    """
    Factory function to get LLM instance.
    
    Args:
        provider: One of 'openai', 'anthropic', 'ollama', 'mock'
        model: Optional model name override
        **kwargs: Provider-specific options
        
    Returns:
        BaseLLM instance
    """
    providers = {
        "openai": (OpenAILLM, "gpt-4"),
        "anthropic": (AnthropicLLM, "claude-3-sonnet-20240229"),
        "ollama": (OllamaLLM, "llama2"),
        "mock": (MockLLM, "mock"),
    }
    
    if provider not in providers:
        raise ValueError(f"Unknown provider: {provider}. Choose from: {list(providers.keys())}")
    
    cls, default_model = providers[provider]
    return cls(model=model or default_model, **kwargs)


# Example usage
if __name__ == "__main__":
    # Test with mock
    llm = get_llm("mock")
    print(f"Using: {llm.name}")
    
    response = llm.generate("What are the zoning constraints for R6?")
    print(f"Response:\n{response}")
