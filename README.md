<div align="center">

# 🏙️ PIMALUOS

**Physics Informed Multi-Agent Land Use Optimization Software**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-40%20passed-brightgreen.svg)]()
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

*AI-powered urban planning with Graph Neural Networks, Multi-Agent RL, and Physics Simulation*

[Documentation](https://pimaluos.github.io/docs) • [Demo](#quick-demo) • [Paper](#citing) • [Dashboard](#dashboard)

</div>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🧠 **Heterogeneous GNN** | 5 edge types for parcel relationships |
| 🤝 **Multi-Agent RL** | 5 stakeholder agents with consensus voting |
| ⚡ **Physics Engine** | Traffic, hydrology, solar simulation |
| 🤖 **LLM-RAG** | Automated zoning constraint extraction |
| 🗺️ **Interactive Dashboard** | deck.gl + CesiumJS visualization |
| 🏛️ **Multi-City** | Manhattan support (expandable architecture) |

---

## 🚀 Quick Start

### Installation & Demo

```bash
# Clone
git clone https://github.com/paryapayami/PIMALUOS.git
cd PIMALUOS

# Create environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install
pip install -e ".[dev]"

# Run a quick 100-parcel demo
python demo.py
```

### Start the Dashboard

```bash
# Terminal 1: Backend
uvicorn pimaluos.api.server:app --reload

# Terminal 2: Frontend
cd dashboard
npm install
npm run dev

# Open http://localhost:3000
```

---

## 💻 Hardware Requirements

### Minimum (Demo - 1,000 parcels)
- **CPU:** 4+ cores (Intel i5/AMD Ryzen 5 or better)
- **RAM:** 8GB minimum, 16GB recommended
- **GPU:** Not required (CPU-only)
- **Storage:** 5GB free space
- **Time:** ~15 minutes for full pipeline

### Full Manhattan (42,075 parcels) ✅ Runs on Consumer Laptop
- **CPU:** Apple M3 Pro, Intel i7, or AMD Ryzen 7 (8+ cores)
- **RAM:** 16GB minimum (tested on 18GB unified memory)
- **GPU:** Not required — entire pipeline runs on CPU
- **Storage:** 5GB free space (results: ~80MB)
- **Time:** ~2.5 hours end-to-end (64 seconds for optimisation with cached training)

### Running the Full-Scale Pipeline
```bash
# Activate environment
source .venv/bin/activate

# Run full 42K-parcel Manhattan pipeline
python run_full_manhattan.py
```

**Output** (in `results/full_manhattan/`):
| File | Description | Size |
|------|-------------|------|
| `manhattan_landuse_plan_42k.csv` | Full plan for 42,075 parcels | 14 MB |
| `manhattan_landuse_plan_42k.geojson` | GeoJSON for GIS/dashboard | 24 MB |
| `cache/gnn_pretrained.pt` | GNN model weights | — |
| `cache/gnn_physics.pt` | Physics-trained GNN weights | — |
| `cache/marl_trainer.pt` | MARL agent weights | — |
| `cache/manhattan_hetero_graph.pt` | Cached heterogeneous graph | — |

The pipeline uses file-based checkpointing (`stage_*.done`). If interrupted, re-running the script resumes from the last completed stage.

**Note:** No GPU, CUDA, or cloud compute is needed. The full 42,075-parcel Manhattan case study is fully reproducible on a consumer laptop.

---

## 🤖 LLM Configuration

PIMALUOS supports three LLM modes for zoning constraint extraction:

### 1. Mock LLM (No API Key Required) ✅
Perfect for testing and development without costs:
```python
from pimaluos.knowledge import get_llm
llm = get_llm('mock')  # Returns pre-defined constraints
```

### 2. Local LLM via Ollama (Free, Private) 🔒
Run models locally without API costs:
```bash
# Install Ollama support
pip install pimaluos[ollama]

# Download and run a model
ollama pull llama2
```

```python
from pimaluos.knowledge import get_llm
llm = get_llm('ollama', model='llama2')
```

### 3. Cloud APIs (OpenAI/Anthropic) ☁️
Requires API keys (costs apply):
- **OpenAI GPT-4:** ~$0.01-0.03 per zoning query
- **Anthropic Claude:** ~$0.015-0.075 per query  
- **Estimated cost for full Manhattan:** $50-150

Set API keys in `.env`:
```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

---

## 📖 Usage

```python
from pimaluos.core import get_data_loader, ParcelGraphBuilder
from pimaluos.models import ParcelGNN, StakeholderAgent
from pimaluos.physics import MultiPhysicsEngine
from pimaluos.knowledge import ConstraintExtractor

# 1. Load city data
loader = get_data_loader("manhattan")
gdf, features = loader.load_and_compute_features()

# 2. Build graph
builder = ParcelGraphBuilder(gdf, features)
graph = builder.build_heterogeneous_graph()

# 3. Train GNN
model = ParcelGNN(in_channels=57, hidden_channels=128)  # 57-dim features
embeddings = model.get_embeddings(graph)

# 4. Extract zoning constraints
extractor = ConstraintExtractor()
constraints = extractor.extract_for_zone('R6')
print(f"Max FAR: {constraints.bulk.max_far}")

# 5. Run physics simulation
physics = MultiPhysicsEngine(gdf)
results = physics.simulate_all(scenario)
```

---

## 🏗️ Architecture

```
pimaluos/
├── core/           # Data loaders, graph builder
├── models/         # GNN, MARL agents, Nash solver
├── knowledge/      # LLM abstraction, RAG pipeline
├── physics/        # Traffic, hydrology, solar
├── api/            # FastAPI server
└── config/         # City-specific settings

dashboard/          # Next.js 14 + deck.gl + Cesium
```

---

## 📊 Dashboard

<div align="center">

| 2D Map View | 3D Digital Twin |
|:-----------:|:---------------:|
| deck.gl with parcel selection | CesiumJS globe view |

</div>

**Features:**
- 🗺️ Interactive parcel selection
- 🎛️ Real-time agent weight adjustment
- 📈 Physics metrics dashboard
- 🔄 WebSocket streaming updates
- 📤 GeoJSON/PDF export

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=pimaluos --cov-report=html
```

---

## 📚 API Reference

| Module | Key Classes |
|--------|-------------|
| `pimaluos.core` | `CityDataLoader`, `ParcelGraphBuilder` |
| `pimaluos.models` | `ParcelGNN`, `StakeholderAgent`, `NashEquilibriumSolver` |
| `pimaluos.knowledge` | `ConstraintExtractor`, `RAGPipeline`, `get_llm()` |
| `pimaluos.physics` | `MultiPhysicsEngine`, `TrafficSimulator` |

---

## 🔑 Environment Variables

Create `.env` in project root:

```bash
# LLM API Keys (optional - mock available for testing)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Dashboard Maps (get free tokens)
NEXT_PUBLIC_MAPBOX_TOKEN=pk.ey...
NEXT_PUBLIC_CESIUM_TOKEN=ey...
```

---

## 📖 Citing

```bibtex
@article{pimaluos2024,
  title={PIMALUOS: Physics Informed Multi-Agent Land Use Optimization},
  author={...},
  journal={Computers, Environment and Urban Systems},
  year={2026}
}
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with ❤️ for urban planners**

[⬆ Back to top](#-pimaluos)

</div>
