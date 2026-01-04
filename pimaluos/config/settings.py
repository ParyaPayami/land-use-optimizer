"""
PIMALUOS Configuration Settings

Centralized configuration management using Pydantic for validation.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Literal
from functools import lru_cache

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class LLMConfig(BaseModel):
    """LLM provider configuration."""
    provider: Literal["openai", "claude", "ollama"] = "openai"
    model: str = "gpt-4"
    temperature: float = 0.0
    max_tokens: int = 4096
    api_key: Optional[str] = None
    base_url: Optional[str] = None  # For Ollama


class CityConfig(BaseModel):
    """City-specific configuration."""
    name: str
    display_name: str
    latitude: float
    longitude: float
    crs: str = "EPSG:4326"
    data_source_url: Optional[str] = None
    zoning_document_paths: List[str] = Field(default_factory=list)
    
    # Edge type configuration for graph building
    edge_types: List[str] = Field(default_factory=lambda: [
        "spatial_adjacency",
        "visual_connectivity", 
        "functional_similarity",
        "infrastructure",
        "regulatory_coupling",
    ])
    
    # Visualization settings
    default_zoom: float = 14.0
    building_color_scheme: str = "land_use"


class PhysicsConfig(BaseModel):
    """Physics simulation configuration."""
    traffic_enabled: bool = True
    hydrology_enabled: bool = True
    solar_enabled: bool = True
    
    # Traffic parameters
    bpr_alpha: float = 0.15
    bpr_beta: float = 4.0
    
    # Hydrology parameters
    design_storm_inches: float = 2.5
    storm_duration_hours: float = 1.0
    
    # Solar parameters
    shadow_threshold_pct: float = 50.0


class Settings(BaseSettings):
    """
    Main application settings.
    
    Configuration priority:
    1. Environment variables (highest)
    2. .env file
    3. Default values
    """
    
    # General
    app_name: str = "PIMALUOS"
    debug: bool = False
    log_level: str = "INFO"
    
    # Data paths
    data_dir: Path = Path("./data")
    models_dir: Path = Path("./models")
    cache_dir: Path = Path("./cache")
    
    # Default city
    default_city: str = "manhattan"
    
    # LLM settings
    llm: LLMConfig = Field(default_factory=LLMConfig)
    
    # Physics settings
    physics: PhysicsConfig = Field(default_factory=PhysicsConfig)
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: List[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    
    # GNN settings
    gnn_hidden_channels: int = 256
    gnn_embed_dim: int = 128
    gnn_num_heads: int = 4
    gnn_dropout: float = 0.2
    
    # MARL settings  
    marl_num_stakeholders: int = 5
    marl_learning_rate: float = 3e-4
    marl_gamma: float = 0.99
    
    class Config:
        env_prefix = "PIMALUOS_"
        env_file = ".env"
        env_nested_delimiter = "__"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def get_city_config(city: str) -> CityConfig:
    """
    Load city-specific configuration from YAML.
    
    Args:
        city: City identifier (e.g., 'manhattan', 'chicago')
        
    Returns:
        CityConfig instance
    """
    config_dir = Path(__file__).parent / "cities"
    config_path = config_dir / f"{city}.yaml"
    
    if not config_path.exists():
        raise ValueError(f"City configuration not found: {city}")
    
    with open(config_path) as f:
        data = yaml.safe_load(f)
    
    return CityConfig(**data)


def get_available_cities() -> List[str]:
    """Get list of available city configurations."""
    config_dir = Path(__file__).parent / "cities"
    return [p.stem for p in config_dir.glob("*.yaml")]
