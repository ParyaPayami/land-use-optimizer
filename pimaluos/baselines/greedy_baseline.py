"""
Greedy Baseline Optimizer

Maximizes single objective (economic value) using greedy hill climbing.
Physics-aware but no multi-stakeholder consideration or RL.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict


class GreedyOptimizer:
    """
    Greedy optimizer that maximizes economic value (FAR × lot_area).
    
    Uses hill climbing: always increase FAR unless it violates constraints
    or reduces economic value.
    
    Args:
        max_iterations: Maximum optimization iterations
    """
    
    def __init__(self, max_iterations: int = 10):
        self.max_iterations = max_iterations
    
    def compute_economic_value(
        self,
        far: np.ndarray,
        lot_area: np.ndarray
    ) -> float:
        """
        Compute total economic value as FAR × lot_area.
        
        Higher FAR on larger lots = more development potential.
        """
        return np.sum(far * lot_area)
    
    def optimize(
        self,
        gdf: pd.DataFrame,
        constraint_masks: pd.DataFrame,
        num_parcels: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Greedy optimization maximizing economic value.
        
        Args:
            gdf: GeoDataFrame with parcel data
            constraint_masks: DataFrame with max_far constraints
            num_parcels: Number of parcels (default: all)
            
        Returns:
            Tuple of (actions array, metrics dict)
        """
        n = num_parcels or len(gdf)
        
        # Get data
        current_far = gdf['built_far'].fillna(1.0).values[:n] if 'built_far' in gdf.columns else np.ones(n)
        lot_area = gdf['lot_area'].fillna(1000).values[:n] if 'lot_area' in gdf.columns else np.ones(n) * 1000
        max_far = constraint_masks['max_far'].values[:n]
        
        # Initialize actions (all maintain)
        actions = np.ones(n, dtype=int)
        
        # Greedy strategy: increase FAR for parcels with highest economic potential
        # Economic potential = (max_far - current_far) × lot_area
        potential_gain = (max_far - current_far) * lot_area
        potential_gain[current_far >= max_far * 0.95] = 0  # Can't increase
        
        # Sort by potential gain (descending)
        sorted_indices = np.argsort(-potential_gain)
        
        # Greedily select parcels to increase
        # Simple strategy: increase top 60% by potential gain
        num_to_increase = int(n * 0.6)
        increase_indices = sorted_indices[:num_to_increase]
        
        # Set actions
        actions[increase_indices] = 2  # Increase
        
        # Small fraction might decrease (bottom 10% by current value)
        current_value = current_far * lot_area
        worst_indices = np.argsort(current_value)[:int(n * 0.1)]
        can_decrease = current_far[worst_indices] > 0.1
        decrease_indices = worst_indices[can_decrease]
        actions[decrease_indices] = 0  # Decrease
        
        # Enforce constraints
        at_max = current_far >= max_far * 0.95
        actions[at_max & (actions == 2)] = 1
        
        at_min = current_far <= 0.1
        actions[at_min & (actions == 0)] = 1
        
        # Calculate final economic value
        # Apply actions to get proposed FAR
        proposed_far = current_far.copy()
        proposed_far[actions == 0] *= 0.8  # Decrease
        proposed_far[actions == 2] *= 1.2  # Increase
        proposed_far = np.clip(proposed_far, 0.1, max_far)
        
        final_value = self.compute_economic_value(proposed_far, lot_area)
        initial_value = self.compute_economic_value(current_far, lot_area)
        
        # Calculate metrics
        metrics = {
            'decrease_count': int(np.sum(actions == 0)),
            'maintain_count': int(np.sum(actions == 1)),
            'increase_count': int(np.sum(actions == 2)),
            'constraint_violations': 0,
            'economic_value_initial': float(initial_value),
            'economic_value_final': float(final_value),
            'economic_improvement': float((final_value - initial_value) / (initial_value + 1e-6)),
        }
        
        return actions, metrics
    
    def __repr__(self):
        return f"GreedyOptimizer(max_iter={self.max_iterations})"
