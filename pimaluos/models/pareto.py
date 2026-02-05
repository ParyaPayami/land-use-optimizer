"""
PIMALUOS Pareto Optimization Module

Implements multi-objective optimization using NSGA-II and NSGA-III
for finding Pareto-optimal land-use configurations.

Objectives:
1. Economic: Maximize development potential and tax revenue
2. Environmental: Minimize environmental impact, maximize green space
3. Social: Maximize affordability and amenity access
4. Equity: Minimize displacement, maximize equitable distribution
"""

from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import pandas as pd
from dataclasses import dataclass
import logging

try:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize
    from pymoo.util.ref_dirs import get_reference_directions
    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False
    logging.warning("pymoo not installed. Install with: pip install pymoo")

logger = logging.getLogger(__name__)


@dataclass
class ParetoSolution:
    """
    Represents a single Pareto-optimal solution.
    
    Attributes:
        land_use_config: Land use configuration (FAR values per parcel)
        objectives: Objective values [economic, environmental, social, equity]
        constraints_satisfied: Whether all constraints are satisfied
        rank: Pareto rank (0 = non-dominated)
        crowding_distance: Crowding distance in objective space
    """
    land_use_config: np.ndarray
    objectives: np.ndarray
    constraints_satisfied: bool
    rank: int
    crowding_distance: float
    
    def __repr__(self):
        return (f"ParetoSolution(rank={self.rank}, "
                f"objectives={self.objectives}, "
                f"feasible={self.constraints_satisfied})")


class LandUseOptimizationProblem(Problem):
    """
    Multi-objective land-use optimization problem for pymoo.
    
    Decision variables: FAR for each parcel
    Objectives: [economic, environmental, social, equity] (to maximize)
    Constraints: Zoning, physics, infrastructure
    """
    
    def __init__(
        self,
        num_parcels: int,
        constraint_masks: pd.DataFrame,
        physics_engine,
        gdf: pd.DataFrame,
        objective_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize optimization problem.
        
        Args:
            num_parcels: Number of parcels to optimize
            constraint_masks: DataFrame with max_far, max_height constraints
            physics_engine: MultiPhysicsEngine for validation
            gdf: GeoDataFrame with parcel data
            objective_weights: Optional weights for objectives
        """
        self.num_parcels = num_parcels
        self.constraint_masks = constraint_masks
        self.physics_engine = physics_engine
        self.gdf = gdf
        
        # Default objective weights
        self.objective_weights = objective_weights or {
            'economic': 1.0,
            'environmental': 1.0,
            'social': 1.0,
            'equity': 1.0
        }
        
        # Get bounds from constraints
        self.xl = np.zeros(num_parcels)  # Minimum FAR = 0
        self.xu = constraint_masks['max_far'].values  # Maximum FAR from zoning
        
        # Initialize pymoo Problem
        # n_var: number of decision variables (parcels)
        # n_obj: number of objectives (4)
        # n_constr: number of constraints (physics violations)
        super().__init__(
            n_var=num_parcels,
            n_obj=4,
            n_constr=3,  # traffic, hydrology, solar
            xl=self.xl,
            xu=self.xu
        )
    
    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evaluate objectives and constraints for population X.
        
        Args:
            X: Population matrix (n_individuals x n_parcels)
            out: Output dict to populate with F (objectives) and G (constraints)
        """
        n_individuals = X.shape[0]
        
        # Initialize output arrays
        F = np.zeros((n_individuals, 4))  # Objectives
        G = np.zeros((n_individuals, 3))  # Constraints
        
        # Evaluate each individual
        for i in range(n_individuals):
            far_values = X[i, :]
            
            # Compute objectives
            F[i, 0] = self._compute_economic_objective(far_values)
            F[i, 1] = self._compute_environmental_objective(far_values)
            F[i, 2] = self._compute_social_objective(far_values)
            F[i, 3] = self._compute_equity_objective(far_values)
            
            # Compute constraints (negative = satisfied)
            G[i, :] = self._compute_constraint_violations(far_values)
        
        # Negate objectives for minimization (pymoo minimizes by default)
        out["F"] = -F
        out["G"] = G
    
    def _compute_economic_objective(self, far_values: np.ndarray) -> float:
        """
        Economic objective: Maximize development potential and tax revenue.
        
        Args:
            far_values: FAR for each parcel
            
        Returns:
            Economic objective value (higher is better)
        """
        # FAR utilization
        max_far = self.constraint_masks['max_far'].values
        far_utilization = np.mean(far_values / (max_far + 1e-6))
        
        # Development potential (additional buildable area)
        lot_areas = self.gdf['lot_area_sqft'].fillna(5000).values
        additional_area = np.sum((far_values - self.gdf.get('far', 0).fillna(0).values) * lot_areas)
        additional_area_normalized = additional_area / (np.sum(lot_areas) + 1e-6)
        
        # Tax revenue proxy (assessed value increase)
        if 'assessed_total' in self.gdf.columns:
            current_value = self.gdf['assessed_total'].fillna(0).values
            value_increase = np.sum(far_values * current_value / (self.gdf.get('far', 1).fillna(1).values + 1e-6))
            value_increase_normalized = value_increase / (np.sum(current_value) + 1e-6)
        else:
            value_increase_normalized = far_utilization
        
        # Weighted combination
        economic_score = (
            0.4 * far_utilization +
            0.3 * additional_area_normalized +
            0.3 * value_increase_normalized
        )
        
        return economic_score
    
    def _compute_environmental_objective(self, far_values: np.ndarray) -> float:
        """
        Environmental objective: Minimize environmental impact.
        
        Args:
            far_values: FAR for each parcel
            
        Returns:
            Environmental objective value (higher is better)
        """
        lot_areas = self.gdf['lot_area_sqft'].fillna(5000).values
        
        # Green space preservation (lower FAR = more green space)
        green_space_score = 1.0 - np.mean(far_values / (self.constraint_masks['max_far'].values + 1e-6))
        
        # Impervious surface (lower is better)
        building_footprints = far_values * lot_areas / 3.0  # Assume 3 floors average
        impervious_ratio = np.sum(building_footprints) / (np.sum(lot_areas) + 1e-6)
        impervious_score = 1.0 - impervious_ratio
        
        # Density gradient (prefer gradual density changes)
        density_variance = np.var(far_values)
        density_score = 1.0 / (1.0 + density_variance)
        
        # Weighted combination
        environmental_score = (
            0.4 * green_space_score +
            0.3 * impervious_score +
            0.3 * density_score
        )
        
        return environmental_score
    
    def _compute_social_objective(self, far_values: np.ndarray) -> float:
        """
        Social objective: Maximize affordability and amenity access.
        
        Args:
            far_values: FAR for each parcel
            
        Returns:
            Social objective value (higher is better)
        """
        # Housing supply (higher FAR in residential = more units)
        if 'land_use' in self.gdf.columns:
            residential_mask = self.gdf['land_use'].str.contains('Residential', na=False)
            residential_far = far_values[residential_mask]
            housing_score = np.mean(residential_far) if len(residential_far) > 0 else 0.5
        else:
            housing_score = 0.5
        
        # Mixed-use development (diversity of land uses)
        far_diversity = 1.0 - (np.std(far_values) / (np.mean(far_values) + 1e-6))
        
        # Amenity access (proxy: density near high-value parcels)
        if 'assessed_total' in self.gdf.columns:
            value_weights = self.gdf['assessed_total'].fillna(0).values
            value_weights = value_weights / (np.sum(value_weights) + 1e-6)
            amenity_score = np.sum(far_values * value_weights)
        else:
            amenity_score = 0.5
        
        # Weighted combination
        social_score = (
            0.4 * housing_score +
            0.3 * far_diversity +
            0.3 * amenity_score
        )
        
        return social_score
    
    def _compute_equity_objective(self, far_values: np.ndarray) -> float:
        """
        Equity objective: Minimize displacement, maximize equitable distribution.
        
        Args:
            far_values: FAR for each parcel
            
        Returns:
            Equity objective value (higher is better)
        """
        # Equitable distribution (low Gini coefficient)
        sorted_far = np.sort(far_values)
        n = len(sorted_far)
        cumsum = np.cumsum(sorted_far)
        gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_far)) / (n * np.sum(sorted_far) + 1e-6) - (n + 1) / n
        equity_distribution = 1.0 - gini
        
        # Displacement risk (avoid large FAR increases in low-value areas)
        if 'assessed_total' in self.gdf.columns and 'far' in self.gdf.columns:
            current_far = self.gdf['far'].fillna(0).values
            far_increase = far_values - current_far
            low_value_mask = self.gdf['assessed_total'] < self.gdf['assessed_total'].median()
            displacement_risk = np.mean(far_increase[low_value_mask]) if np.any(low_value_mask) else 0
            displacement_score = 1.0 / (1.0 + displacement_risk)
        else:
            displacement_score = 0.5
        
        # Opportunity distribution (development in underserved areas)
        opportunity_score = 1.0 - np.std(far_values) / (np.mean(far_values) + 1e-6)
        
        # Weighted combination
        equity_score = (
            0.4 * equity_distribution +
            0.3 * displacement_score +
            0.3 * opportunity_score
        )
        
        return equity_score
    
    def _compute_constraint_violations(self, far_values: np.ndarray) -> np.ndarray:
        """
        Compute constraint violations for physics constraints.
        
        Args:
            far_values: FAR for each parcel
            
        Returns:
            Array of constraint violations [traffic, hydrology, solar]
            Negative values = satisfied, positive = violated
        """
        violations = np.zeros(3)
        
        try:
            # Create land use scenario
            scenario = {}
            for i, far in enumerate(far_values):
                scenario[i] = {
                    'use': 'mixed',
                    'units': int(far * 10),
                    'floor_area': far * self.gdf.iloc[i].get('lot_area_sqft', 5000),
                    'lot_area_sqft': self.gdf.iloc[i].get('lot_area_sqft', 5000),
                    'height_ft': min(far * 12, self.constraint_masks.iloc[i]['max_height_ft'])
                }
            
            # Run physics simulation
            results = self.physics_engine.simulate_all(scenario)
            
            # Traffic constraint (congestion < 1.5)
            violations[0] = results['traffic']['avg_congestion_ratio'] - 1.5
            
            # Hydrology constraint (capacity utilization < 1.0)
            violations[1] = results['hydrology']['capacity_utilization'] - 1.0
            
            # Solar constraint (shadow < 50%)
            violations[2] = results['solar']['avg_shadow_pct'] / 100.0 - 0.5
            
        except Exception as e:
            logger.warning(f"Physics simulation failed: {e}")
            # Conservative: assume constraints are violated
            violations[:] = 0.1
        
        return violations


class ParetoOptimizer:
    """
    Multi-objective Pareto optimization for land-use planning.
    
    Uses NSGA-II or NSGA-III to find Pareto-optimal solutions
    balancing economic, environmental, social, and equity objectives.
    """
    
    def __init__(
        self,
        num_parcels: int,
        constraint_masks: pd.DataFrame,
        physics_engine,
        gdf: pd.DataFrame,
        algorithm: str = 'nsga2'
    ):
        """
        Initialize Pareto optimizer.
        
        Args:
            num_parcels: Number of parcels to optimize
            constraint_masks: Zoning constraints
            physics_engine: Physics simulation engine
            gdf: Parcel GeoDataFrame
            algorithm: 'nsga2' or 'nsga3'
        """
        if not PYMOO_AVAILABLE:
            raise ImportError("pymoo is required for Pareto optimization. "
                            "Install with: pip install pymoo")
        
        self.num_parcels = num_parcels
        self.constraint_masks = constraint_masks
        self.physics_engine = physics_engine
        self.gdf = gdf
        self.algorithm = algorithm.lower()
        
        # Create optimization problem
        self.problem = LandUseOptimizationProblem(
            num_parcels=num_parcels,
            constraint_masks=constraint_masks,
            physics_engine=physics_engine,
            gdf=gdf
        )
        
        logger.info(f"ParetoOptimizer initialized with {algorithm.upper()}")
    
    def optimize(
        self,
        population_size: int = 100,
        num_generations: int = 100,
        seed: int = 42
    ) -> List[ParetoSolution]:
        """
        Run Pareto optimization.
        
        Args:
            population_size: Population size
            num_generations: Number of generations
            seed: Random seed
            
        Returns:
            List of Pareto-optimal solutions
        """
        logger.info(f"Starting {self.algorithm.upper()} optimization...")
        logger.info(f"  Population: {population_size}")
        logger.info(f"  Generations: {num_generations}")
        
        # Configure algorithm
        if self.algorithm == 'nsga2':
            algorithm = NSGA2(
                pop_size=population_size,
                eliminate_duplicates=True
            )
        elif self.algorithm == 'nsga3':
            # Reference directions for 4 objectives
            ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=12)
            algorithm = NSGA3(
                ref_dirs=ref_dirs,
                pop_size=population_size,
                eliminate_duplicates=True
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        # Run optimization
        result = minimize(
            self.problem,
            algorithm,
            ('n_gen', num_generations),
            seed=seed,
            verbose=True
        )
        
        logger.info(f"Optimization complete!")
        logger.info(f"  Pareto solutions found: {len(result.F)}")
        
        # Convert to ParetoSolution objects
        solutions = []
        for i in range(len(result.F)):
            solution = ParetoSolution(
                land_use_config=result.X[i],
                objectives=-result.F[i],  # Negate back (we minimized negatives)
                constraints_satisfied=np.all(result.G[i] <= 0) if result.G is not None else True,
                rank=0,  # All returned solutions are non-dominated
                crowding_distance=0.0  # Would need to compute from result
            )
            solutions.append(solution)
        
        return solutions
    
    def find_knee_solution(self, solutions: List[ParetoSolution]) -> ParetoSolution:
        """
        Find the "knee" solution - best compromise across objectives.
        
        Uses normalized Euclidean distance from ideal point.
        
        Args:
            solutions: List of Pareto solutions
            
        Returns:
            Knee solution
        """
        if not solutions:
            raise ValueError("No solutions provided")
        
        # Extract objective values
        objectives = np.array([s.objectives for s in solutions])
        
        # Normalize objectives to [0, 1]
        obj_min = objectives.min(axis=0)
        obj_max = objectives.max(axis=0)
        objectives_norm = (objectives - obj_min) / (obj_max - obj_min + 1e-6)
        
        # Ideal point (all objectives = 1)
        ideal = np.ones(4)
        
        # Find solution closest to ideal
        distances = np.linalg.norm(objectives_norm - ideal, axis=1)
        knee_idx = np.argmin(distances)
        
        logger.info(f"Knee solution found at index {knee_idx}")
        logger.info(f"  Objectives: {solutions[knee_idx].objectives}")
        
        return solutions[knee_idx]
    
    def export_pareto_front(
        self,
        solutions: List[ParetoSolution],
        output_path: str
    ):
        """
        Export Pareto front to CSV.
        
        Args:
            solutions: List of Pareto solutions
            output_path: Output file path
        """
        data = []
        for i, sol in enumerate(solutions):
            data.append({
                'solution_id': i,
                'economic': sol.objectives[0],
                'environmental': sol.objectives[1],
                'social': sol.objectives[2],
                'equity': sol.objectives[3],
                'feasible': sol.constraints_satisfied,
                'rank': sol.rank
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        logger.info(f"Pareto front exported to {output_path}")


# Convenience function
def optimize_land_use_pareto(
    num_parcels: int,
    constraint_masks: pd.DataFrame,
    physics_engine,
    gdf: pd.DataFrame,
    algorithm: str = 'nsga2',
    population_size: int = 100,
    num_generations: int = 100
) -> Tuple[List[ParetoSolution], ParetoSolution]:
    """
    Convenience function for Pareto optimization.
    
    Args:
        num_parcels: Number of parcels
        constraint_masks: Zoning constraints
        physics_engine: Physics engine
        gdf: Parcel GeoDataFrame
        algorithm: 'nsga2' or 'nsga3'
        population_size: Population size
        num_generations: Number of generations
        
    Returns:
        Tuple of (all_solutions, knee_solution)
    """
    optimizer = ParetoOptimizer(
        num_parcels=num_parcels,
        constraint_masks=constraint_masks,
        physics_engine=physics_engine,
        gdf=gdf,
        algorithm=algorithm
    )
    
    solutions = optimizer.optimize(
        population_size=population_size,
        num_generations=num_generations
    )
    
    knee = optimizer.find_knee_solution(solutions)
    
    return solutions, knee
