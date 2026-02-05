"""
PIMALUOS Baseline Optimizers

This module provides simple baseline methods for land-use optimization
to compare against the full PIMALUOS approach.

Available baselines:
- RandomOptimizer: Random action selection
- RuleBasedOptimizer: Heuristic rules based on current FAR
- GreedyOptimizer: Single-objective (economic) maximization
"""

from pimaluos.baselines.random_baseline import RandomOptimizer
from pimaluos.baselines.rule_based_baseline import RuleBasedOptimizer
from pimaluos.baselines.greedy_baseline import GreedyOptimizer

__all__ = [
    'RandomOptimizer',
    'RuleBasedOptimizer',
    'GreedyOptimizer',
]
