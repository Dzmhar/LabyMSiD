"""
Simulated Annealing (SA) algorithm implementation.

SA is a metaheuristic that simulates the annealing process in metallurgy.
It starts with a high temperature and gradually cools down, allowing the algorithm
to escape local optima by occasionally accepting worse solutions.
"""

import numpy as np
from typing import Tuple, Optional

from core.algorithms.base import MetaheuristicAlgorithm


class SimulatedAnnealing(MetaheuristicAlgorithm):
    """
    Simulated Annealing algorithm for the Load Balancing problem.

    Components:
    - Initial solution: Randomly generated solution
    - Neighborhood: Swap or change one task's server assignment
    - Temperature schedule: Geometric cooling
    - Acceptance criterion: Metropolis criterion
    """

    def __init__(
        self,
        problem,
        max_iterations: int = 100,
        initial_temperature: float = 1000.0,
        cooling_rate: float = 0.95,
        seed: Optional[int] = None,
    ):
        """
        Initialize Simulated Annealing.

        Args:
            problem: The LoadBalancingProblem instance.
            max_iterations: Maximum number of iterations.
            initial_temperature: Starting temperature.
            cooling_rate: Cooling rate for geometric cooling.
            seed: Random seed.
        """
        super().__init__(problem, max_iterations, seed)
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate

    @property
    def name(self) -> str:
        return "Simulated Annealing"

    def _get_temperature(self, iteration: int) -> float:
        """
        Calculate temperature at given iteration using geometric cooling.

        T(t) = T0 * alpha^t

        Args:
            iteration: Current iteration number.

        Returns:
            Current temperature.
        """
        return self.initial_temperature * (self.cooling_rate ** iteration)

    def _generate_neighbor(self, solution: np.ndarray) -> np.ndarray:
        """
        Generate a neighbor solution by modifying the current solution.

        Neighborhood generation:
        - With 50% probability: change one random task's server to a random server
        - With 50% probability: swap servers of two random tasks

        Args:
            solution: Current solution.

        Returns:
            Neighbor solution.
        """
        neighbor = solution.copy()
        num_tasks = self.problem.num_tasks
        num_servers = self.problem.num_servers

        if self._rng.random() < 0.5:
            task_idx = self._rng.randint(num_tasks)
            new_server = self._rng.randint(num_servers)
            neighbor[task_idx] = new_server
        else:
            i, j = self._rng.choice(num_tasks, 2, replace=False)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

        return neighbor

    def _acceptance_probability(self, old_score: float, new_score: float, temperature: float) -> float:
        """
        Calculate acceptance probability using Metropolis criterion.

        P(accept) = 1 if delta < 0
        P(accept) = exp(-delta/T) if delta >= 0

        where delta = new_score - old_score

        Args:
            old_score: Current solution score.
            new_score: Neighbor solution score.
            temperature: Current temperature.

        Returns:
            Acceptance probability.
        """
        delta = new_score - old_score

        if delta < 0:
            return 1.0
        else:
            return np.exp(-delta / temperature)

    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        Run the Simulated Annealing optimization.

        Returns:
            A tuple (best_solution, best_score).
        """
        self.best_solution = None
        self.best_score = float('inf')
        self.convergence_history = []

        current_solution = self.initialize_solution()
        current_score = self.evaluate(current_solution)

        self.update_best(current_solution, current_score)
        self.convergence_history.append(self.best_score)

        for iteration in range(self.max_iterations):
            temperature = self._get_temperature(iteration)

            neighbor = self._generate_neighbor(current_solution)
            neighbor_score = self.evaluate(neighbor)

            acceptance_prob = self._acceptance_probability(current_score, neighbor_score, temperature)

            if self._rng.random() < acceptance_prob:
                current_solution = neighbor
                current_score = neighbor_score

                self.update_best(current_solution, current_score)

            self.convergence_history.append(self.best_score)

        return self.best_solution, self.best_score