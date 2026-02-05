"""
Rule-Based Baseline Optimizer

Uses simple heuristic rules to decide FAR changes based on current utilization.
More sophisticated than random but doesn't use ML or multi-objective optimization.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict


class RuleBasedOptimizer:
    """
    Rule-based baseline using FAR utilization heuristics.
    
    Rules:
    - If current_far < 0.7 * max_far → increase (underdeveloped)
    - If current_far > 0.9 * max_far → decrease (overdeveloped)
    - Otherwise → maintain
    
    Args:
        low_threshold: Threshold for underdevelopment (default: 0.7)
        high_threshold: Threshold for overdevelopment (default: 0.9)
    """
    
    def __init__(
        self, 
        low_threshold: float = 0.7,
        high_threshold: float = 0.9
    ):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
    
    def optimize(
        self,
        gdf: pd.DataFrame,
        constraint_masks: pd.DataFrame,
        num_parcels: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Apply rule-based optimization.
        
        Args:
            gdf: GeoDataFrame with parcel data
            constraint_masks: DataFrame with max_far constraints
            num_parcels: Number of parcels (default: all)
            
        Returns:
            Tuple of (actions array, metrics dict)
        """
        n = num_parcels or len(gdf)
        
        # Get current FAR and max FAR
        current_far = gdf['built_far'].fillna(1.0).values[:n] if 'built_far' in gdf.columns else np.ones(n)
        max_far = constraint_masks['max_far'].values[:n]
        
        # Calculate utilization ratio
        utilization = current_far / (max_far + 1e-6)
        
        # Apply rules
        actions = np.ones(n, dtype=int)  # Default: maintain
        
        # Rule 1: Increase if underdeveloped
        underdeveloped = utilization < self.low_threshold
        actions[underdeveloped] = 2
        
        # Rule 2: Decrease if overdeveloped (rare, but possible with variance)
        overdeveloped = utilization > self.high_threshold
        actions[overdeveloped] = 0
        
        # Additional rule: Consider zone type
        # If residential zone and high density, maintain or decrease
        if 'zone_district' in gdf.columns:
            zone = gdf['zone_district'].values[:n]
            is_residential = pd.Series(zone).str.contains('R', na=False).values
            high_density_residential = is_residential & (utilization > 0.8)
            actions[high_density_residential] = np.minimum(actions[high_density_residential], 1)
        
        # Enforce hard constraints
        at_max = current_far >= max_far * 0.95
        actions[at_max & (actions == 2)] = 1
        
        at_min = current_far <= 0.1
        actions[at_min & (actions == 0)] = 1
        
        # Calculate metrics
        metrics = {
            'decrease_count': int(np.sum(actions == 0)),
            'maintain_count': int(np.sum(actions == 1)),
            'increase_count': int(np.sum(actions == 2)),
            'constraint_violations': 0,
            'avg_utilization': float(np.mean(utilization)),
            'underdeveloped_count': int(np.sum(underdeveloped)),
            'overdeveloped_count': int(np.sum(overdeveloped)),
        }
        
        return actions, metrics
    
    def __repr__(self):
        return f"RuleBasedOptimizer(low={self.low_threshold}, high={self.high_threshold})"
