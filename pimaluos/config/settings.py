"""
PIMALUOS configuration.

City configurations live in ``pimaluos/config/cities/*.yaml``. Only ``yaml`` and
``pydantic`` are required; there is no dependency on ``pydantic-settings``.
"""

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field


class CityConfig(BaseModel):
    """City-specific configuration loaded from YAML."""

    model_config = ConfigDict(extra="allow")

    name: str
    display_name: str
    latitude: float
    longitude: float
    # Projected CRS (units must be feet or metres; see ``crs_units_ft``).
    crs: str = "EPSG:2263"
    crs_units_ft: bool = True
    # Reference point (e.g. CBD) in WGS84 used for the distance-to-centre feature.
    center_lon: Optional[float] = None
    center_lat: Optional[float] = None
    edge_types: List[str] = Field(
        default_factory=lambda: [
            "spatial_adjacency",
            "proximity",
            "functional_similarity",
            "street_frontage",
            "regulatory_coupling",
        ]
    )


class Settings(BaseModel):
    """Global defaults. Override by passing a dict to the CLI ``--config`` file."""

    data_dir: Path = Path("./data")
    results_dir: Path = Path("./results")
    default_city: str = "manhattan"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


def get_city_config(city: str) -> CityConfig:
    """Load a city configuration from ``config/cities/<city>.yaml``."""
    config_path = Path(__file__).parent / "cities" / f"{city}.yaml"
    if not config_path.exists():
        raise ValueError(f"City configuration not found: {city}")
    with open(config_path) as f:
        data: Dict = yaml.safe_load(f)
    geo = data.get("geographic", {})
    data.setdefault("crs", geo.get("crs", "EPSG:2263"))
    return CityConfig(**data)


def get_available_cities() -> List[str]:
    config_dir = Path(__file__).parent / "cities"
    return sorted(p.stem for p in config_dir.glob("*.yaml"))
