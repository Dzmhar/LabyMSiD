"""
Bees Algorithm (BA) implementation.

BA is a metaheuristic inspired by the foraging behavior of honey bees.
Scout bees search for food sources, and then recruiter bees search
around the best sources.
"""

import numpy as np
from typing import Tuple, Optional, List

from core.algorithms.base import MetaheuristicAlgorithm


class BeesAlgorithm(MetaheuristicAlgorithm):
    """
    Bees Algorithm for the Load Balancing problem.

    Components:
    - Random solution generation: Scout bees generate random solutions
    - Selection of best sites: Select elite and other selected sites based on fitness
    - Neighborhood search: Send bees to search around selected sites
    """

    def __init__(
        self,
        problem,
        max_iterations: int = 100,
        num_scouts: int = 20,
        num_elite_sites: int = 3,
        num_selected_sites: int = 6,
        num_bees_elite: int = 10,
        num_bees_selected: int = 5,
        neighborhood_size: float = 0.5,
        seed: Optional[int] = None,
    ):
        """
        Initialize Bees Algorithm.

        Args:
            problem: The LoadBalancingProblem instance.
            max_iterations: Maximum number of iterations.
            num_scouts: Number of scout bees.
            num_elite_sites: Number of elite sites to select.
            num_selected_sites: Total number of selected sites (including elite).
            num_bees_elite: Number of bees sent to elite sites.
            num_bees_selected: Number of bees sent to selected (non-elite) sites.
            neighborhood_size: Radius of neighborhood search.
            seed: Random seed.
        """
        super().__init__(problem, max_iterations, seed)
        self.num_scouts = num_scouts
        self.num_elite_sites = num_elite_sites
        self.num_selected_sites = num_selected_sites
        self.num_bees_elite = num_bees_elite
        self.num_bees_selected = num_bees_selected
        self.neighborhood_size = neighborhood_size

        self.solutions: List[np.ndarray] = []
        self.scores: List[float] = []

    @property
    def name(self) -> str:
        return "Bees Algorithm"

    def _generate_random_solutions(self) -> None:
        """
        Generate random solutions (scout bees).
        """
        self.solutions = []
        self.scores = []

        for _ in range(self.num_scouts):
            solution = self.initialize_solution()
            score = self.evaluate(solution)

            self.solutions.append(solution)
            self.scores.append(score)

            self.update_best(solution, score)

    def _select_sites(self) -> List[int]:
        """
        Select sites based on fitness (sorted by score, lower is better).

        Returns:
            List of indices of selected sites.
        """
        sorted_indices = np.argsort(self.scores)
        selected = sorted_indices[: self.num_selected_sites].tolist()

        return selected

    def _calculate_neighborhood_modification(self) -> int:
        """
        Calculate how many tasks to modify in neighborhood search.

        Returns:
            Number of tasks to modify.
        """
        neighborhood_tasks = max(
            1, int(self.problem.num_tasks * self.neighborhood_size)
        )
        return neighborhood_tasks

    def _neighborhood_search(self, solution: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Perform neighborhood search around a solution.

        Randomly modifies a subset of tasks to nearby servers.

        Args:
            solution: Original solution to search around.

        Returns:
            A tuple (new_solution, new_score).
        """
        neighbor = solution.copy()
        num_tasks_to_modify = self._calculate_neighborhood_modification()

        indices = self._rng.choice(
            self.problem.num_tasks, num_tasks_to_modify, replace=False
        )

        for idx in indices:
            current_server = neighbor[idx]
            available_servers = [
                s for s in range(self.problem.num_servers) if s != current_server
            ]
            if available_servers:
                new_server = self._rng.choice(available_servers)
                neighbor[idx] = new_server

        score = self.evaluate(neighbor)
        return neighbor, score

    def _search_around_sites(self, site_indices: List[int]) -> None:
        """
        Send bees to search around selected sites.

        Args:
            site_indices: Indices of selected sites.
        """
        elite_indices = site_indices[: self.num_elite_sites]
        selected_indices = site_indices[self.num_elite_sites : self.num_selected_sites]

        for idx in elite_indices:
            site_solution = self.solutions[idx]

            for _ in range(self.num_bees_elite - 1):
                if self._rng.random() < 0.5:
                    neighbor, score = self._neighborhood_search(site_solution)
                else:
                    neighbor = self.initialize_solution()
                    score = self.evaluate(neighbor)

                self.solutions.append(neighbor)
                self.scores.append(score)

                self.update_best(neighbor, score)

        for idx in selected_indices:
            site_solution = self.solutions[idx]

            for _ in range(self.num_bees_selected - 1):
                if self._rng.random() < 0.5:
                    neighbor, score = self._neighborhood_search(site_solution)
                else:
                    neighbor = self.initialize_solution()
                    score = self.evaluate(neighbor)

                self.solutions.append(neighbor)
                self.scores.append(score)

                self.update_best(neighbor, score)

    def _trim_population(self) -> None:
        """
        Trim the population back to num_scouts best solutions.
        """
        sorted_indices = np.argsort(self.scores)

        self.solutions = [self.solutions[i] for i in sorted_indices[: self.num_scouts]]
        self.scores = [self.scores[i] for i in sorted_indices[: self.num_scouts]]

    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        Run the Bees Algorithm optimization.

        Returns:
            A tuple (best_solution, best_score).
        """
        self.best_solution = None
        self.best_score = float('inf')
        self.convergence_history = []

        self._generate_random_solutions()
        self.convergence_history.append(self.best_score)

        for _ in range(self.max_iterations):
            site_indices = self._select_sites()

            self._search_around_sites(site_indices)

            self._trim_population()

            self.convergence_history.append(self.best_score)

        return self.best_solution, self.best_score