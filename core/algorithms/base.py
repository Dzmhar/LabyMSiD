"""
Base class for metaheuristic algorithms.

This module provides an abstract base class that all metaheuristic algorithms
must inherit from, defining the common interface.
"""

from abc import ABC, abstractmethod
import numpy as np
import time
from typing import Tuple, Optional


class MetaheuristicAlgorithm(ABC):
    """
    Abstract base class for metaheuristic algorithms.

    All algorithms must implement the optimize method and provide a way to
    generate initial solutions and track convergence history.
    """

    def __init__(self, problem, max_iterations: int = 100, seed: Optional[int] = None):
        """
        Initialize the algorithm.

        Args:
            problem: The LoadBalancingProblem instance to solve.
            max_iterations: Maximum number of iterations.
            seed: Optional random seed for reproducibility.
        """
        self.problem = problem
        self.max_iterations = max_iterations
        self.seed = seed

        self.best_solution: Optional[np.ndarray] = None
        self.best_score: float = float('inf')
        self.convergence_history: list[float] = []

        self._rng = np.random.RandomState(seed)

    @abstractmethod
    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        Run the optimization algorithm.

        Returns:
            A tuple (best_solution, best_score).
        """
        pass

    def initialize_solution(self) -> np.ndarray:
        """
        Generate an initial solution.

        Returns:
            A random initial solution.
        """
        return self.problem.generate_random_solution()

    def evaluate(self, solution: np.ndarray) -> float:
        """
        Evaluate a solution.

        Args:
            solution: The solution to evaluate.

        Returns:
            The makespan score.
        """
        return self.problem.calculate_makespan(solution)

    def update_best(self, solution: np.ndarray, score: float) -> None:
        """
        Update the best solution if the new score is better.

        Args:
            solution: The solution to check.
            score: The score for the solution.
        """
        if score < self.best_score:
            self.best_score = score
            self.best_solution = solution.copy()

    def run_with_timing(self) -> Tuple[np.ndarray, float, float]:
        """
        Run the optimization and measure execution time.

        Returns:
            A tuple (best_solution, best_score, execution_time_seconds).
        """
        start_time = time.perf_counter()
        solution, score = self.optimize()
        end_time = time.perf_counter()
        execution_time = end_time - start_time

        return solution, score, execution_time

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the algorithm name."""
        pass