"""
PIMALUOS - Physics Informed Multi-Agent Land Use Optimization Software

A comprehensive platform for urban planners, developers, and decision makers
to optimize land use using Graph Neural Networks, Multi-Agent Reinforcement
Learning, and physics-based simulation.

Modules:
    - core: Data loading and graph building
    - models: GNN and MARL agent implementations
    - knowledge: LLM-RAG legal constraint parsing
    - physics: Traffic, hydrology, and solar simulation
    - api: FastAPI backend server
"""

__version__ = "0.1.0"
__author__ = "PIMALUOS Team"

# Core imports
from pimaluos.core.data_loader import CityDataLoader, ManhattanDataLoader, get_data_loader
from pimaluos.core.graph_builder import ParcelGraphBuilder

# Model imports
from pimaluos.models.gnn import ParcelGNN, HeterogeneousGAT
from pimaluos.models.agents import StakeholderAgent, MultiAgentEnvironment

# Physics imports
from pimaluos.physics.engine import MultiPhysicsEngine
from pimaluos.physics.digital_twin import UrbanDigitalTwin

# System integration
from pimaluos.system import UrbanOptSystem

__all__ = [
    # Core
    "CityDataLoader",
    "ManhattanDataLoader",
    "get_data_loader",
    "ParcelGraphBuilder",
    # Models
    "ParcelGNN",
    "HeterogeneousGAT",
    "StakeholderAgent",
    "MultiAgentEnvironment",
    # Physics
    "MultiPhysicsEngine",
    "UrbanDigitalTwin",
    # System
    "UrbanOptSystem",
]
