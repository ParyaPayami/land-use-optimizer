"""
Random Baseline Optimizer

Randomly selects actions (decrease/maintain/increase FAR) for each parcel
while respecting zoning constraints. Used as a naive baseline.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict


class RandomOptimizer:
    """
    Random baseline that assigns FAR changes uniformly at random.
    
    Ensures constraint compliance by checking max FAR limits.
    
    Args:
        random_seed: Random seed for reproducibility
    """
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)
    
    def optimize(
        self,
        gdf: pd.DataFrame,
        constraint_masks: pd.DataFrame,
        num_parcels: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Generate random actions for parcels.
        
        Args:
            gdf: GeoDataFrame with parcel data
            constraint_masks: DataFrame with max_far, max_height constraints
            num_parcels: Number of parcels (default: all)
            
        Returns:
            Tuple of (actions array, metrics dict)
        """
        n = num_parcels or len(gdf)
        
        # Get current FAR and max FAR
        current_far = gdf['built_far'].fillna(1.0).values[:n] if 'built_far' in gdf.columns else np.ones(n)
        max_far = constraint_masks['max_far'].values[:n]
        
        # Action space: 0=decrease, 1=maintain, 2=increase
        actions = self.rng.randint(0, 3, size=n)
        
        # Enforce constraints: can't increase if already at max
        at_max = current_far >= max_far * 0.95
        actions[at_max & (actions == 2)] = 1  # Change increase to maintain
        
        # Can't decrease if already at minimum (e.g., 0.1)
        at_min = current_far <= 0.1
        actions[at_min & (actions == 0)] = 1  # Change decrease to maintain
        
        # Calculate metrics
        metrics = {
            'decrease_count': int(np.sum(actions == 0)),
            'maintain_count': int(np.sum(actions == 1)),
            'increase_count': int(np.sum(actions == 2)),
            'constraint_violations': 0,  # By construction
        }
        
        return actions, metrics
    
    def __repr__(self):
        return f"RandomOptimizer(seed={self.random_seed})"
