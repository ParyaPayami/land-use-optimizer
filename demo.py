#!/usr/bin/env python3
"""
PIMALUOS Demo Script

Interactive demonstration of the Physics Informed Multi-Agent
Land Use Optimization Software capabilities.

Run: python demo.py
"""

import sys
import time

def print_header(text: str):
    """Print formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")

def print_step(step: int, text: str):
    """Print step indicator."""
    print(f"\n[Step {step}] {text}")
    print("-" * 40)

def demo_imports():
    """Demo: Import all modules."""
    print_step(1, "Importing PIMALUOS modules...")
    
    import pimaluos
    print(f"✓ pimaluos v{pimaluos.__version__}")
    
    from pimaluos.core import get_data_loader, ParcelGraphBuilder
    print("✓ Core module: Data loaders, Graph builder")
    
    from pimaluos.models import ParcelGNN, StakeholderAgent, NashEquilibriumSolver
    print("✓ Models module: GNN, MARL agents, Nash solver")
    
    from pimaluos.knowledge import get_llm, ConstraintExtractor
    print("✓ Knowledge module: LLM abstraction, RAG pipeline")
    
    from pimaluos.physics import MultiPhysicsEngine, TrafficSimulator
    print("✓ Physics module: Traffic, Hydrology, Solar simulation")
    
    return True

def demo_constraint_extraction():
    """Demo: LLM-RAG constraint extraction."""
    print_step(2, "Extracting zoning constraints with LLM-RAG...")
    
    from pimaluos.knowledge import get_llm, ConstraintExtractor
    
    # Use mock LLM for demo (no API key needed)
    llm = get_llm('mock')
    print(f"  Using: {llm.name}")
    
    extractor = ConstraintExtractor()
    
    zones = ['R6', 'C4', 'M1']
    for zone in zones:
        constraints = extractor.extract_for_zone(zone)
        print(f"\n  Zone {zone}:")
        print(f"    Max FAR: {constraints.bulk.max_far}")
        print(f"    Max Height: {constraints.bulk.max_height_ft} ft")
        print(f"    Type: {constraints.zone_type}")
    
    return True

def demo_physics_simulation():
    """Demo: Multi-physics simulation."""
    print_step(3, "Running multi-physics simulation...")
    
    import geopandas as gpd
    from shapely.geometry import box
    from pimaluos.physics import MultiPhysicsEngine
    
    # Create sample parcels
    parcels = [box(i * 100, 0, (i + 1) * 100, 100) for i in range(5)]
    gdf = gpd.GeoDataFrame({
        'geometry': parcels,
        'lot_area_sqft': [5000] * 5,
    }, crs='EPSG:4326')
    
    engine = MultiPhysicsEngine(gdf)
    
    # Simulate scenario
    scenario = {
        i: {
            'use': 'residential',
            'units': 10,
            'floor_area': 5000,
            'lot_area_sqft': 5000,
            'height_ft': 35 + i * 10,
        }
        for i in range(5)
    }
    
    print("  Scenario: 5 residential parcels with varying heights")
    
    results = engine.simulate_all(scenario)
    
    print(f"\n  Traffic Results:")
    print(f"    Avg Congestion: {results['traffic']['avg_congestion_ratio']:.2f}")
    
    print(f"\n  Hydrology Results:")
    print(f"    Runoff: {results['hydrology']['peak_runoff_cfs']:.1f} cfs")
    print(f"    Capacity: {results['hydrology']['capacity_utilization']:.1%}")
    
    print(f"\n  Solar Results:")
    print(f"    Shadow Impact: {results['solar']['avg_shadow_pct']:.1f}%")
    
    print(f"\n  Violations: {results['violations']['total_violations']}")
    
    return True

def demo_agents():
    """Demo: Multi-agent stakeholder system."""
    print_step(4, "Creating stakeholder agents...")
    
    from pimaluos.models import StakeholderAgent
    from pimaluos.models.agents import ConsensusVotingMechanism
    import torch
    
    agent_types = ['resident', 'developer', 'planner', 'environmentalist', 'equity_advocate']
    agents = {}
    
    for agent_type in agent_types:
        agent = StakeholderAgent(
            state_dim=128,
            action_dim=3,
            agent_type=agent_type
        )
        agents[agent_type] = agent
        print(f"  ✓ {agent_type.title()}: awareness = {agent.awareness}")
    
    # Voting mechanism
    print("\n  Testing consensus voting...")
    voting = ConsensusVotingMechanism(voting_strategy='weighted')
    
    # Simulate actions
    actions = {
        'resident': [0, 1, 2, 1, 1],
        'developer': [2, 2, 2, 2, 0],
        'planner': [1, 1, 1, 1, 1],
        'environmentalist': [0, 0, 1, 0, 0],
        'equity_advocate': [1, 1, 0, 1, 2],
    }
    
    consensus = voting.aggregate_votes(actions)
    print(f"  Consensus actions: {consensus}")
    
    return True

def demo_api():
    """Demo: FastAPI server endpoints."""
    print_step(5, "Testing API endpoints...")
    
    from pimaluos.api.server import create_app
    
    app = create_app()
    
    routes = [r.path for r in app.routes if hasattr(r, 'path')]
    print("  Available endpoints:")
    for route in routes[:10]:
        print(f"    {route}")
    
    print(f"\n  Total endpoints: {len(routes)}")
    print("\n  To start server: uvicorn pimaluos.api.server:app --reload")
    
    return True

def main():
    """Run the demo."""
    print_header("PIMALUOS DEMO")
    print("Physics Informed Multi-Agent Land Use Optimization Software")
    print("Version 0.1.0")
    
    demos = [
        ("Module Imports", demo_imports),
        ("Constraint Extraction", demo_constraint_extraction),
        ("Physics Simulation", demo_physics_simulation),
        ("Multi-Agent System", demo_agents),
        ("API Server", demo_api),
    ]
    
    results = []
    for name, func in demos:
        try:
            success = func()
            results.append((name, "✅ PASS"))
        except Exception as e:
            print(f"\n❌ Error: {e}")
            results.append((name, "❌ FAIL"))
    
    print_header("DEMO COMPLETE")
    
    print("Results:")
    for name, status in results:
        print(f"  {status} {name}")
    
    print("\n" + "=" * 60)
    print("  Next Steps:")
    print("  1. Start API:   uvicorn pimaluos.api.server:app")
    print("  2. Dashboard:   cd dashboard && npm run dev")
    print("  3. Open:        http://localhost:3000")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
