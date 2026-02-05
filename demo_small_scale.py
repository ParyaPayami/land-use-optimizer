#!/usr/bin/env python3
"""
Small-scale reproducible demo for consumer hardware.

This demo uses a subset of 100 parcels to demonstrate the complete
PIMALUOS pipeline in a reasonable time on consumer hardware.

Hardware requirements:
- CPU: Any modern CPU (2+ cores)
- RAM: 8 GB minimum, 16 GB recommended
- GPU: Not required for this demo
- Time: ~5-10 minutes

Results are saved to: ./results/small_scale_demo/
"""

import sys
import time
from pathlib import Path
import json
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./results/small_scale_demo/demo.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Run small-scale reproducible demo."""
    
    # Create results directory
    results_dir = Path('./results/small_scale_demo')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("  PIMALUOS SMALL-SCALE DEMO")
    logger.info("  Reproducible Demo for Consumer Hardware")
    logger.info("=" * 70)
    logger.info("")
    
    # Record start time
    start_time = time.time()
    
    # Configuration
    config = {
        'city': 'manhattan',
        'data_subset_size': 100,  # Small subset for demo
        'llm_mode': 'mock',
        'device': 'cpu',
        'random_seed': 42,
        'gnn_epochs': 10,  # Reduced for demo
        'physics_epochs': 5,  # Reduced for demo
        'marl_iterations': 20,  # Reduced for demo
        'marl_steps_per_iteration': 10
    }
    
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    logger.info("")
    
    try:
        from pimaluos import UrbanOptSystem
        
        # Step 1: Initialize system
        logger.info("Step 1/7: Initializing UrbanOptSystem...")
        system = UrbanOptSystem(
            city=config['city'],
            data_subset_size=config['data_subset_size'],
            llm_mode=config['llm_mode'],
            device=config['device'],
            random_seed=config['random_seed']
        )
        logger.info("✓ System initialized\n")
        
        # Step 2: Load data and build graph
        logger.info("Step 2/7: Loading data and building graph...")
        gdf, features = system.load_data()
        graph = system.build_graph()
        logger.info(f"✓ Loaded {len(gdf)} parcels with {features.shape[1]} features\n")
        
        # Step 3: Extract constraints
        logger.info("Step 3/7: Extracting zoning constraints...")
        constraints = system.extract_constraints()
        logger.info(f"✓ Extracted constraints for {len(constraints)} parcels\n")
        
        # Step 4: Pre-train GNN
        logger.info(f"Step 4/7: Pre-training GNN ({config['gnn_epochs']} epochs)...")
        history_pretrain = system.pretrain_gnn(num_epochs=config['gnn_epochs'])
        logger.info(f"✓ GNN pre-trained (final loss: {history_pretrain['pretrain_losses'][-1]:.4f})\n")
        
        # Step 5: Train with physics feedback
        logger.info(f"Step 5/7: Training with physics feedback ({config['physics_epochs']} epochs)...")
        history_physics = system.train_with_physics_feedback(num_epochs=config['physics_epochs'])
        logger.info(f"✓ Physics-informed training complete (final loss: {history_physics['physics_losses'][-1]:.4f})\n")
        
        # Step 6: Optimize with MARL
        logger.info(f"Step 6/7: Optimizing with MARL ({config['marl_iterations']} iterations)...")
        trainer = system.optimize_with_marl(
            num_iterations=config['marl_iterations'],
            steps_per_iteration=config['marl_steps_per_iteration']
        )
        logger.info("✓ MARL optimization complete\n")
        
        # Step 7: Generate final plan
        logger.info("Step 7/7: Generating final land-use plan...")
        final_plan = system.generate_final_plan(
            trainer,
            output_path=results_dir / 'final_plan.csv'
        )
        logger.info(f"✓ Final plan generated for {len(final_plan)} parcels\n")
        
        # Save checkpoint
        logger.info("Saving checkpoint...")
        system.save_checkpoint(results_dir / 'checkpoint.pth')
        logger.info(f"✓ Checkpoint saved to {results_dir / 'checkpoint.pth'}\n")
        
        # Save results summary
        elapsed = time.time() - start_time
        results = {
            'config': config,
            'data': {
                'num_parcels': len(gdf),
                'num_features': features.shape[1],
                'num_constraints': len(constraints)
            },
            'training': {
                'pretrain_final_loss': float(history_pretrain['pretrain_losses'][-1]),
                'physics_final_loss': float(history_physics['physics_losses'][-1]),
                'num_pretrain_epochs': len(history_pretrain['pretrain_losses']),
                'num_physics_epochs': len(history_physics['physics_losses'])
            },
            'actions': {
                'decrease_far': int(sum(final_plan['action'] == 0)),
                'maintain_far': int(sum(final_plan['action'] == 1)),
                'increase_far': int(sum(final_plan['action'] == 2))
            },
            'performance': {
                'total_time_seconds': elapsed,
                'total_time_formatted': f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
            }
        }
        
        with open(results_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("=" * 70)
        logger.info("  DEMO COMPLETE!")
        logger.info("=" * 70)
        logger.info(f"\nResults saved to: {results_dir}")
        logger.info(f"  - final_plan.csv: Land-use plan for {len(final_plan)} parcels")
        logger.info(f"  - checkpoint.pth: Model checkpoint")
        logger.info(f"  - results.json: Summary statistics")
        logger.info(f"  - demo.log: Execution log")
        logger.info(f"\nExecution time: {results['performance']['total_time_formatted']}")
        logger.info(f"\nAction distribution:")
        logger.info(f"  Decrease FAR: {results['actions']['decrease_far']} parcels")
        logger.info(f"  Maintain FAR: {results['actions']['maintain_far']} parcels")
        logger.info(f"  Increase FAR: {results['actions']['increase_far']} parcels")
        logger.info("")
        
        return True
        
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
