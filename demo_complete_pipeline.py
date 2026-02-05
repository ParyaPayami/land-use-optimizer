#!/usr/bin/env python3
"""
Demo script showing the complete PIMALUOS pipeline using UrbanOptSystem.

This script demonstrates the workflow described in the manuscript.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from pimaluos import UrbanOptSystem


def main():
    """Run the complete PIMALUOS pipeline demo."""
    
    print("=" * 70)
    print("  PIMALUOS COMPLETE PIPELINE DEMO")
    print("  Physics-Informed Multi-Agent Land Use Optimization")
    print("=" * 70)
    print()
    
    # Initialize system with small subset for demo
    print("Step 1: Initializing UrbanOptSystem...")
    system = UrbanOptSystem(
        city='manhattan',
        data_subset_size=None,  # Full Manhattan dataset (~42K parcels)
        llm_mode='mock',  # No API key needed
        device='cpu'  # Use CPU for compatibility
    )
    print("✓ System initialized\n")
    
    # Load data and build graph
    print("Step 2: Loading data and building graph...")
    system.load_data()
    system.build_graph()
    print("✓ Graph built\n")
    
    # Extract constraints
    print("Step 3: Extracting zoning constraints...")
    system.extract_constraints()
    print("✓ Constraints extracted\n")
    
    # Pre-train GNN
    print("Step 4: Pre-training GNN (50 epochs)...")
    system.pretrain_gnn(num_epochs=50)
    print("✓ GNN pre-trained\n")
    
    # Train with physics feedback
    print("Step 5: Training with physics feedback (20 epochs)...")
    system.train_with_physics_feedback(num_epochs=20)
    print("✓ Physics-informed training complete\n")
    
    # Optimize with MARL
    print("Step 6: Optimizing with multi-agent RL (100 iterations)...")
    trainer = system.optimize_with_marl(
        num_iterations=100,
        steps_per_iteration=50
    )
    print("✓ MARL optimization complete\n")
    
    # Generate final plan
    print("Step 7: Generating final land-use plan...")
    # Ensure directory exists
    Path('results/full_scale_simulation').mkdir(parents=True, exist_ok=True)
    
    final_plan = system.generate_final_plan(
        trainer,
        output_path=Path('results/full_scale_simulation/manhattan_landuse.csv')
    )
    print("✓ Final plan generated\n")
    
    # Save checkpoint
    print("Step 8: Saving checkpoint...")
    system.save_checkpoint(Path('./output/checkpoint.pth'))
    print("✓ Checkpoint saved\n")
    
    # Summary
    print("=" * 70)
    print("  PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\nResults:")
    print(f"  - Parcels optimized: {len(final_plan)}")
    print(f"  - Final plan saved to: results/full_scale_simulation/manhattan_landuse.csv")
    print(f"  - Checkpoint saved to: ./output/checkpoint.pth")
    print(f"\nLand Use Distribution:")
    if 'proposed_use_label' in final_plan.columns:
        counts = final_plan['proposed_use_label'].value_counts()
        for label, count in counts.items():
            print(f"  - {label}: {count} parcels ({count/len(final_plan)*100:.1f}%)")
    print()
    print("Next steps:")
    print("  1. Run Dashboard: streamlit run dashboard_app.py")
    print("  2. Select 'Proposed Land Use' to view optimization results")
    print()


if __name__ == "__main__":
    # Create output directory
    Path('./output').mkdir(exist_ok=True)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
