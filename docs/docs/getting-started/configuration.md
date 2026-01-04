# Configuration

PIMALUOS uses a hierarchical configuration system with Pydantic models.

## Environment Variables

Set these in your shell or `.env` file:

```bash
# LLM API Keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Dashboard (optional)
export NEXT_PUBLIC_MAPBOX_TOKEN="pk.ey..."
export NEXT_PUBLIC_CESIUM_TOKEN="ey..."
```

## City Configuration

City-specific settings are stored in YAML files under `pimaluos/config/cities/`:

```yaml
# manhattan.yaml
name: Manhattan
display_name: "Manhattan, NYC"
state: NY
country: USA

geographic:
  bounds: [-74.047, 40.679, -73.907, 40.882]
  center: [-73.985, 40.748]
  default_zoom: 14
  crs: "EPSG:2263"  # NY State Plane

data_sources:
  parcels:
    url: "https://data.cityofnewyork.us/api/geospatial/..."
    format: geojson
  zoning:
    url: "https://data.cityofnewyork.us/api/geospatial/..."
    format: geojson

land_use_categories:
  - residential
  - commercial
  - industrial
  - mixed
  - open_space
```

## Loading Configuration

```python
from pimaluos.config import get_city_config, Settings

# Get city-specific config
config = get_city_config("manhattan")
print(config.name)  # Manhattan
print(config.geographic.center)  # [-73.985, 40.748]

# Global settings
settings = Settings()
print(settings.gnn.hidden_dim)  # 128
```

## Settings Schema

```python
class Settings:
    # LLM Settings
    llm_provider: str = "openai"
    llm_model: str = "gpt-4"
    llm_temperature: float = 0.0
    
    # Physics Settings
    traffic_alpha: float = 0.15
    traffic_beta: float = 4.0
    design_rainfall_intensity: float = 2.5
    min_solar_exposure: float = 0.6
    
    # GNN Settings
    gnn_hidden_dim: int = 128
    gnn_num_layers: int = 3
    gnn_dropout: float = 0.2
    
    # MARL Settings
    marl_learning_rate: float = 3e-4
    marl_gamma: float = 0.99
    marl_gae_lambda: float = 0.95
```

## Overriding Settings

Settings can be overridden via environment variables:

```bash
export PIMALUOS_GNN_HIDDEN_DIM=256
export PIMALUOS_LLM_PROVIDER=anthropic
```

Or programmatically:

```python
from pimaluos.config import Settings

settings = Settings(
    gnn={"hidden_dim": 256},
    llm_provider="anthropic"
)
```
