# Installation

## Requirements

- Python 3.9+
- CUDA-capable GPU (recommended for GNN training)
- 16GB+ RAM (for large parcel datasets)

## Installation Methods

### From PyPI (Recommended)

```bash
pip install pimaluos
```

### From Source

```bash
git clone https://github.com/pimaluos/pimaluos.git
cd pimaluos
pip install -e ".[dev]"
```

### Optional Dependencies

```bash
# For documentation
pip install pimaluos[docs]

# For local LLM support (Ollama)
pip install pimaluos[ollama]

# All development dependencies
pip install pimaluos[dev]
```

## Configuration

### Environment Variables

Create a `.env` file in your project root:

```bash
# LLM API Keys (at least one required for constraint extraction)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Optional: for dashboard map tokens
NEXT_PUBLIC_MAPBOX_TOKEN=pk.ey...
NEXT_PUBLIC_CESIUM_TOKEN=ey...
```

### Verify Installation

```python
import pimaluos
print(f"PIMALUOS v{pimaluos.__version__}")

# Test core imports
from pimaluos.core import CityDataLoader, ParcelGraphBuilder
from pimaluos.models import ParcelGNN, StakeholderAgent
from pimaluos.physics import MultiPhysicsEngine

print("All modules loaded successfully!")
```

## Next Steps

- [Quick Start Tutorial](quickstart.md)
- [Configuration Guide](configuration.md)
