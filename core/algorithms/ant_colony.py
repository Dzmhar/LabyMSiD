"""
Ant Colony Optimization (ACO) algorithm implementation.

ACO is a metaheuristic inspired by the foraging behavior of ants.
Ants communicate through pheromones and use them to construct solutions.
"""

import numpy as np
from typing import Tuple, Optional

from core.algorithms.base import MetaheuristicAlgorithm


class AntColonyOptimization(MetaheuristicAlgorithm):
    """
    Ant Colony Optimization for the Load Balancing problem.

    Components:
    - Solution construction: Ants build solutions based on pheromone levels
    - Pheromone matrix: Tracks the quality of task-server assignments
    - Pheromone update: Evaporation and deposit based on solution quality

    Note: This ACO variant treats each task independently, assigning it to a server
    based on pheromone levels and heuristic information (inverse of average load).
    """

    def __init__(
        self,
        problem,
        max_iterations: int = 100,
        num_ants: int = 20,
        alpha: float = 1.0,
        beta: float = 2.0,
        evaporation_rate: float = 0.5,
        q: float = 100.0,
        seed: Optional[int] = None,
    ):
        """
        Initialize Ant Colony Optimization.

        Args:
            problem: The LoadBalancingProblem instance.
            max_iterations: Maximum number of iterations.
            num_ants: Number of ants.
            alpha: Pheromone importance weight.
            beta: Heuristic importance weight.
            evaporation_rate: Pheromone evaporation rate.
            q: Pheromone deposit coefficient.
            seed: Random seed.
        """
        super().__init__(problem, max_iterations, seed)
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.q = q

        self.pheromone: Optional[np.ndarray] = None
        self.best_solutions: list = []

    @property
    def name(self) -> str:
        return "Ant Colony Optimization"

    def _initialize_pheromone(self) -> None:
        """
        Initialize pheromone matrix with uniform values.
        """
        self.pheromone = np.ones(
            (self.problem.num_tasks, self.problem.num_servers), dtype=np.float64
        )

    def _construct_solution(self) -> Tuple[np.ndarray, float]:
        """
        Construct a solution using the ant colony scheme.

        Each ant constructs a solution by:
        1. For each task, selecting a server probabilistically based on pheromone
           and heuristic (inverse load) information

        Args:
            None.

        Returns:
            A tuple (solution, score).
        """
        solution = np.zeros(self.problem.num_tasks, dtype=np.int64)
        server_loads = np.zeros(self.problem.num_servers, dtype=np.float64)

        for task_id in range(self.problem.num_tasks):
            probabilities = self._calculate_server_probabilities(task_id, server_loads)

            server = self._rng.choice(self.problem.num_servers, p=probabilities)
            solution[task_id] = server
            server_loads[server] += self.problem.task_times[task_id]

        score = self.evaluate(solution)
        return solution, score

    def _calculate_server_probabilities(
        self, task_id: int, current_loads: np.ndarray
    ) -> np.ndarray:
        """
        Calculate selection probabilities for a task.

        Formula: P(i->j) = (tau_ij^alpha * eta_ij^beta) / sum(tau_ik^alpha * eta_ik^beta)

        where:
        - tau_ij is pheromone level for task i to server j
        - eta_ij is heuristic (1 / (current_load + avg_task_time))

        Args:
            task_id: The current task ID.
            current_loads: Current load of each server.

        Returns:
            Probability array for selecting each server.
        """
        pheromone_levels = self.pheromone[task_id]

        avg_task_time = np.mean(self.problem.task_times)
        heuristic = 1.0 / (current_loads + avg_task_time + 1e-6)

        desirability = (pheromone_levels ** self.alpha) * (heuristic ** self.beta)

        total = desirability.sum()
        if total > 0:
            probabilities = desirability / total
        else:
            probabilities = np.ones(self.problem.num_servers) / self.problem.num_servers

        return probabilities

    def _update_pheromone(self, solutions: list, scores: list) -> None:
        """
        Update pheromone levels based on solution quality.

        Pheromone update:
        1. Evaporation: tau = (1 - rho) * tau
        2. Deposit: tau = tau + Q / score for each ant's solution

        Args:
            solutions: List of solutions constructed by ants.
            scores: List of corresponding scores.
        """
        self.pheromone *= (1.0 - self.evaporation_rate)

        best_idx = np.argmin(scores)
        best_score = scores[best_idx]

        deposit = self.q / best_score

        for task_id, server_id in enumerate(solutions[best_idx]):
            self.pheromone[task_id, server_id] += deposit

    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        Run the Ant Colony Optimization.

        Returns:
            A tuple (best_solution, best_score).
        """
        self.best_solution = None
        self.best_score = float('inf')
        self.convergence_history = []

        self._initialize_pheromone()
        self.convergence_history.append(self.best_score)

        for _ in range(self.max_iterations):
            solutions = []
            scores = []

            for _ in range(self.num_ants):
                solution, score = self._construct_solution()
                solutions.append(solution)
                scores.append(score)

                self.update_best(solution, score)

            self._update_pheromone(solutions, scores)
            self.convergence_history.append(self.best_score)

        return self.best_solution, self.best_score