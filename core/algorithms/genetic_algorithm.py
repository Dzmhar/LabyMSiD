"""
Genetic Algorithm (GA) implementation.

GA is a population-based metaheuristic inspired by natural selection.
It uses selection, crossover, and mutation operators to evolve solutions over generations.
"""

import numpy as np
from typing import Tuple, Optional, List

from core.algorithms.base import MetaheuristicAlgorithm


class GeneticAlgorithm(MetaheuristicAlgorithm):
    """
    Genetic Algorithm for the Load Balancing problem.

    Components:
    - Population generation: Random initial population
    - Selection: Tournament selection
    - Crossover: Single-point crossover
    - Mutation: Random server reassignment
    """

    def __init__(
        self,
        problem,
        max_iterations: int = 100,
        population_size: int = 50,
        crossover_prob: float = 0.8,
        mutation_prob: float = 0.1,
        tournament_size: int = 3,
        seed: Optional[int] = None,
    ):
        """
        Initialize Genetic Algorithm.

        Args:
            problem: The LoadBalancingProblem instance.
            max_iterations: Maximum number of generations.
            population_size: Size of the population.
            crossover_prob: Crossover probability.
            mutation_prob: Mutation probability per gene.
            tournament_size: Tournament size for selection.
            seed: Random seed.
        """
        super().__init__(problem, max_iterations, seed)
        self.population_size = population_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size

        self.population: List[np.ndarray] = []
        self.fitness: List[float] = []

    @property
    def name(self) -> str:
        return "Genetic Algorithm"

    def _initialize_population(self) -> None:
        """
        Initialize the population with random solutions.
        """
        self.population = []
        self.fitness = []

        for _ in range(self.population_size):
            solution = self.initialize_solution()
            score = self.evaluate(solution)

            self.population.append(solution)
            self.fitness.append(score)

            self.update_best(solution, score)

    def _tournament_selection(self) -> np.ndarray:
        """
        Perform tournament selection.

        Select the best individual from a random tournament.

        Returns:
            Selected individual.
        """
        tournament_indices = self._rng.choice(
            self.population_size, self.tournament_size, replace=False
        )

        best_idx = min(tournament_indices, key=lambda i: self.fitness[i])

        return self.population[best_idx].copy()

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform single-point crossover.

        Args:
            parent1: First parent.
            parent2: Second parent.

        Returns:
            Two offspring.
        """
        if self._rng.random() < self.crossover_prob:
            crossover_point = self._rng.randint(1, self.problem.num_tasks)

            child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])

            return child1, child2
        else:
            return parent1.copy(), parent2.copy()

    def _mutate(self, solution: np.ndarray) -> np.ndarray:
        """
        Perform mutation by randomly changing some genes.

        Args:
            solution: Solution to mutate.

        Returns:
            Mutated solution.
        """
        mutated = solution.copy()

        for i in range(len(mutated)):
            if self._rng.random() < self.mutation_prob:
                mutated[i] = self._rng.randint(self.problem.num_servers)

        return mutated

    def _create_offspring(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create offspring through selection, crossover, and mutation.

        Returns:
            Two offspring.
        """
        parent1 = self._tournament_selection()
        parent2 = self._tournament_selection()

        child1, child2 = self._crossover(parent1, parent2)

        child1 = self._mutate(child1)
        child2 = self._mutate(child2)

        return child1, child2

    def _replace_population(self, offspring: List[np.ndarray]) -> None:
        """
        Replace the old population with new offspring.

        Uses (mu + lambda) replacement strategy.

        Args:
            offspring: List of offspring solutions.
        """
        combined_solutions = self.population + offspring
        combined_fitness = self.fitness + [self.evaluate(sol) for sol in offspring]

        indices = np.argsort(combined_fitness)[:self.population_size]

        self.population = [combined_solutions[i] for i in indices]
        self.fitness = [combined_fitness[i] for i in indices]

    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        Run the Genetic Algorithm optimization.

        Returns:
            A tuple (best_solution, best_score).
        """
        self.best_solution = None
        self.best_score = float('inf')
        self.convergence_history = []

        self._initialize_population()
        self.convergence_history.append(self.best_score)

        for _ in range(self.max_iterations):
            offspring = []

            for _ in range(self.population_size // 2):
                child1, child2 = self._create_offspring()
                offspring.append(child1)
                offspring.append(child2)

            self._replace_population(offspring)

            self.update_best(self.population[0], self.fitness[0])
            self.convergence_history.append(self.best_score)

        return self.best_solution, self.best_score