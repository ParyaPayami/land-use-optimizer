"""
PIMALUOS Nash Equilibrium Solver

Game-theoretic solver for multi-agent land-use optimization:
- Pure and mixed Nash equilibrium computation
- Iterative best response dynamics
- Shapley values for fair benefit distribution
- Price of Anarchy analysis
"""

from typing import Dict, List, Tuple, Optional, Any
from itertools import product, combinations
import numpy as np
import pandas as pd


class NashEquilibriumSolver:
    """
    Computes Nash equilibrium for multi-agent land-use game.
    
    A Nash equilibrium is a configuration where no agent can
    improve their utility by unilaterally changing their action.
    
    Args:
        agents: Dict of stakeholder agents
        environment: MultiAgentEnvironment instance
        num_parcels: Number of parcels in the game
    """
    
    def __init__(
        self, 
        agents: Dict,
        environment,
        num_parcels: int = 100
    ):
        self.agents = agents
        self.env = environment
        self.num_parcels = num_parcels
        self.action_space = 3  # decrease, maintain, increase
    
    def compute_payoff_matrix(
        self, 
        parcel_idx: int,
        agent_types: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        Compute payoff matrix for a single parcel.
        
        For each action combination, compute expected utility for each agent.
        
        Returns:
            Dict mapping agent_type to payoff matrix
        """
        num_agents = len(agent_types)
        payoff_matrices = {}
        
        # Initial state
        state = self.env.reset()
        
        # All action combinations
        action_combinations = list(product(range(self.action_space), repeat=num_agents))
        
        for agent_idx, agent_type in enumerate(agent_types):
            shape = [self.action_space] * num_agents
            payoff_matrix = np.zeros(shape)
            
            for actions in action_combinations:
                # Create action dict
                action_dict = {
                    agent_types[i]: [
                        actions[i] if j == parcel_idx else 1  # maintain for other parcels
                        for j in range(self.num_parcels)
                    ]
                    for i in range(num_agents)
                }
                
                # Evaluate
                _, rewards, _, _ = self.env.step(action_dict)
                payoff_matrix[actions] = rewards[agent_type]
                
                # Reset
                self.env.reset()
            
            payoff_matrices[agent_type] = payoff_matrix
        
        return payoff_matrices
    
    def find_pure_nash_equilibrium(
        self,
        payoff_matrices: Dict[str, np.ndarray],
        agent_types: List[str]
    ) -> List[Tuple]:
        """
        Find pure strategy Nash equilibria.
        
        Returns:
            List of equilibrium action profiles
        """
        nash_equilibria = []
        
        for actions in product(range(self.action_space), repeat=len(agent_types)):
            is_equilibrium = True
            
            for agent_idx, agent_type in enumerate(agent_types):
                current_payoff = payoff_matrices[agent_type][actions]
                
                # Check if any deviation improves payoff
                for alt_action in range(self.action_space):
                    if alt_action == actions[agent_idx]:
                        continue
                    
                    alt_actions = list(actions)
                    alt_actions[agent_idx] = alt_action
                    alt_payoff = payoff_matrices[agent_type][tuple(alt_actions)]
                    
                    if alt_payoff > current_payoff:
                        is_equilibrium = False
                        break
                
                if not is_equilibrium:
                    break
            
            if is_equilibrium:
                nash_equilibria.append(actions)
        
        return nash_equilibria
    
    def find_mixed_nash_2player(
        self,
        payoff_1: np.ndarray,
        payoff_2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find mixed strategy Nash equilibrium for 2-player game.
        
        Uses support enumeration via nashpy library (if available)
        or falls back to uniform distribution.
        
        Returns:
            Tuple of strategy distributions for each player
        """
        try:
            import nashpy as nash
            game = nash.Game(payoff_1, payoff_2)
            equilibria = list(game.support_enumeration())
            
            if equilibria:
                return equilibria[0]
        except ImportError:
            pass
        
        # Fallback: uniform distribution
        return (
            np.ones(payoff_1.shape[0]) / payoff_1.shape[0],
            np.ones(payoff_2.shape[0]) / payoff_2.shape[0]
        )
    
    def find_equilibrium_iterative(
        self,
        agent_types: List[str] = None,
        max_iterations: int = 100,
        tolerance: float = 1e-4
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Find Nash equilibrium using iterative best response dynamics.
        
        Each agent sequentially plays best response to others' strategies.
        
        Returns:
            Tuple of (equilibrium_config, info)
        """
        if agent_types is None:
            agent_types = list(self.agents.keys())
        
        print("\n" + "=" * 60)
        print("NASH EQUILIBRIUM SEARCH (Iterative Best Response)")
        print("=" * 60)
        print(f"Agents: {agent_types}")
        print(f"Parcels: {self.num_parcels}")
        
        # Initialize with "maintain" action for all
        current_actions = {
            agent_type: [1] * self.num_parcels
            for agent_type in agent_types
        }
        
        state = self.env.reset()
        converged = False
        
        for iteration in range(max_iterations):
            old_actions = {k: v.copy() for k, v in current_actions.items()}
            
            # Each agent plays best response
            for agent_type in agent_types:
                for parcel_idx in range(self.num_parcels):
                    best_action = 1
                    best_utility = float("-inf")
                    
                    for action in range(self.action_space):
                        # Evaluate action
                        test_actions = {k: v.copy() for k, v in current_actions.items()}
                        test_actions[agent_type][parcel_idx] = action
                        
                        _, rewards, _, _ = self.env.step(test_actions)
                        
                        if rewards[agent_type] > best_utility:
                            best_utility = rewards[agent_type]
                            best_action = action
                        
                        self.env.reset()
                    
                    current_actions[agent_type][parcel_idx] = best_action
            
            # Check convergence
            changed = sum(
                1 for at in agent_types
                for i in range(self.num_parcels)
                if old_actions[at][i] != current_actions[at][i]
            )
            
            if changed == 0:
                converged = True
                print(f"\n✓ Converged in {iteration + 1} iterations")
                break
            
            if (iteration + 1) % 20 == 0:
                print(f"  Iteration {iteration + 1}: {changed} changes")
        
        if not converged:
            print(f"\n⚠ Did not converge after {max_iterations} iterations")
        
        # Get final state
        final_state, _, _, info = self.env.step(current_actions)
        final_far = self.env.current_far.cpu().numpy() if hasattr(self.env, 'current_far') else np.ones(self.num_parcels)
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            "parcel_id": range(self.num_parcels),
            "equilibrium_far": final_far,
        })
        
        # Add agent actions
        for agent_type in agent_types:
            result_df[f"{agent_type}_action"] = current_actions[agent_type]
        
        # Consensus action (majority vote)
        result_df["consensus_action"] = [
            max(set([current_actions[at][i] for at in agent_types]),
                key=[current_actions[at][i] for at in agent_types].count)
            for i in range(self.num_parcels)
        ]
        
        return result_df, info
    
    def compute_social_welfare(self, config_df: pd.DataFrame) -> float:
        """Compute total social welfare (sum of all agent utilities)."""
        # Simplified: sum of equilibrium FARs normalized
        return config_df["equilibrium_far"].mean()


class ShapleyValueCalculator:
    """
    Compute Shapley values for fair benefit distribution.
    
    Shapley value measures each agent's average marginal contribution
    across all possible coalition orderings.
    """
    
    def __init__(self, agents: Dict, environment):
        self.agents = agents
        self.env = environment
        self.agent_types = list(agents.keys())
    
    def characteristic_function(self, coalition: List[str]) -> float:
        """
        Compute value achievable by a coalition.
        
        Args:
            coalition: List of cooperating agent types
            
        Returns:
            Coalition's achievable value
        """
        if not coalition:
            return 0.0
        
        state = self.env.reset()
        
        # Coalition maximizes joint utility
        best_value = float("-inf")
        
        # Determine agent types to iterate over
        agent_keys = self.env.agents.keys() if hasattr(self.env, 'agents') else self.agent_types
        
        for _ in range(20):  # Random search
            num_parcels = getattr(self.env, 'num_parcels', 10)
            actions = {
                at: [np.random.choice(3) for _ in range(num_parcels)]
                for at in agent_keys
            }
            
            _, rewards, _, _ = self.env.step(actions)
            
            coalition_value = sum(rewards.get(at, 0) for at in coalition)
            best_value = max(best_value, coalition_value)
            
            self.env.reset()
        
        return best_value
    
    def compute_shapley_values(self) -> Dict[str, float]:
        """
        Compute Shapley value for each agent.
        
        Returns:
            Dict mapping agent_type to Shapley value
        """
        n = len(self.agent_types)
        shapley_values = {at: 0.0 for at in self.agent_types}
        
        for agent in self.agent_types:
            marginal_contributions = []
            others = [at for at in self.agent_types if at != agent]
            
            # All subsets not containing agent
            for r in range(n):
                for coalition in combinations(others, r):
                    coalition = list(coalition)
                    
                    v_with = self.characteristic_function(coalition + [agent])
                    v_without = self.characteristic_function(coalition)
                    
                    marginal_contributions.append(v_with - v_without)
            
            shapley_values[agent] = np.mean(marginal_contributions) if marginal_contributions else 0.0
        
        print("\n" + "=" * 60)
        print("SHAPLEY VALUES (Fair Benefit Distribution)")
        print("=" * 60)
        for agent, value in shapley_values.items():
            print(f"  {agent}: {value:.4f}")
        
        return shapley_values


class ParetoAnalyzer:
    """
    Analyze Pareto efficiency and price of anarchy.
    """
    
    def __init__(self, nash_solver: NashEquilibriumSolver):
        self.solver = nash_solver
    
    def compute_pareto_frontier(
        self,
        num_samples: int = 100
    ) -> List[Dict[str, float]]:
        """
        Approximate Pareto frontier via random sampling.
        
        Returns:
            List of non-dominated utility vectors
        """
        agent_types = list(self.solver.agents.keys())
        samples = []
        
        for _ in range(num_samples):
            # Random configuration
            actions = {
                at: [np.random.choice(3) for _ in range(self.solver.num_parcels)]
                for at in agent_types
            }
            
            _, rewards, _, _ = self.solver.env.step(actions)
            samples.append(rewards.copy())
            self.solver.env.reset()
        
        # Find non-dominated points
        pareto = []
        for point in samples:
            dominated = False
            for other in samples:
                if all(other[at] >= point[at] for at in agent_types) and \
                   any(other[at] > point[at] for at in agent_types):
                    dominated = True
                    break
            if not dominated:
                pareto.append(point)
        
        return pareto
    
    def price_of_anarchy(
        self,
        nash_config: pd.DataFrame,
        pareto_welfare: float
    ) -> Dict[str, float]:
        """
        Compute price of anarchy: ratio of Nash vs optimal welfare.
        
        Returns:
            Dict with welfare comparisons
        """
        nash_welfare = self.solver.compute_social_welfare(nash_config)
        
        poa = nash_welfare / pareto_welfare if pareto_welfare > 0 else 1.0
        
        return {
            "nash_welfare": nash_welfare,
            "pareto_welfare": pareto_welfare,
            "price_of_anarchy": poa,
            "efficiency_loss_pct": (1 - poa) * 100,
        }


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("NASH EQUILIBRIUM SOLVER TEST")
    print("=" * 60)
    
    # Would require actual environment and agents to run
    print("\nNash equilibrium solver loaded successfully!")
    print("Requires trained agents and environment for full test.")
