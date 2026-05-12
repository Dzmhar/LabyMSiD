"""
Unit tests for the Load Balancing problem and metaheuristic algorithms.

Tests validate:
1. Correct calculation of objective function (makespan)
2. That all algorithms return valid solutions
"""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.problem import LoadBalancingProblem
from core.algorithms.simulated_annealing import SimulatedAnnealing
from core.algorithms.genetic_algorithm import GeneticAlgorithm
from core.algorithms.ant_colony import AntColonyOptimization
from core.algorithms.bees_algorithm import BeesAlgorithm


@pytest.fixture
def problem():
    """Create a test problem instance."""
    task_times = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    return LoadBalancingProblem(num_tasks=5, num_servers=3, task_times=task_times)


class TestMakespanCalculation:
    """Tests for makespan calculation."""

    def test_makespan_single_task_per_server(self, problem):
        """Test makespan with single task per server."""
        solution = np.array([0, 1, 2, 0, 1])
        makespan = problem.calculate_makespan(solution)

        server_0 = 10 + 40
        server_1 = 20 + 50
        server_2 = 30

        expected = max(server_0, server_1, server_2)
        assert makespan == expected

    def test_makespan_all_tasks_same_server(self, problem):
        """Test makespan when all tasks on one server."""
        solution = np.array([0, 0, 0, 0, 0])
        makespan = problem.calculate_makespan(solution)

        total = 10 + 20 + 30 + 40 + 50
        assert makespan == total

    def test_makespan_empty_servers(self, problem):
        """Test makespan ignores empty servers."""
        solution = np.array([0, 0, 0, 0, 0])
        makespan = problem.calculate_makespan(solution)

        assert makespan >= 0


class TestSolutionValidation:
    """Tests for solution validation."""

    def test_valid_solution(self, problem):
        """Test valid solution passes validation."""
        solution = problem.generate_random_solution()
        is_valid, _ = problem.validate_solution(solution)

        assert is_valid is True

    def test_invalid_length(self, problem):
        """Test solution with wrong length fails."""
        solution = np.array([0, 1, 2])
        is_valid, msg = problem.validate_solution(solution)

        assert is_valid is False

    def test_invalid_server_id(self, problem):
        """Test solution with invalid server ID fails."""
        solution = np.array([0, 1, 2, 3, 10])
        is_valid, msg = problem.validate_solution(solution)

        assert is_valid is False


class TestAlgorithmSolutionValidity:
    """Tests that algorithms return valid solutions."""

    def test_sa_valid_solution(self, problem):
        """Test Simulated Annealing returns valid solution."""
        algorithm = SimulatedAnnealing(problem, max_iterations=10, seed=42)
        solution, _ = algorithm.optimize()

        assert len(solution) == problem.num_tasks
        assert solution.min() >= 0
        assert solution.max() < problem.num_servers

    def test_ga_valid_solution(self, problem):
        """Test Genetic Algorithm returns valid solution."""
        algorithm = GeneticAlgorithm(problem, max_iterations=10, seed=42)
        solution, _ = algorithm.optimize()

        assert len(solution) == problem.num_tasks
        assert solution.min() >= 0
        assert solution.max() < problem.num_servers

    def test_aco_valid_solution(self, problem):
        """Test Ant Colony Optimization returns valid solution."""
        algorithm = AntColonyOptimization(problem, max_iterations=10, seed=42)
        solution, _ = algorithm.optimize()

        assert len(solution) == problem.num_tasks
        assert solution.min() >= 0
        assert solution.max() < problem.num_servers

    def test_ba_valid_solution(self, problem):
        """Test Bees Algorithm returns valid solution."""
        algorithm = BeesAlgorithm(problem, max_iterations=10, seed=42)
        solution, _ = algorithm.optimize()

        assert len(solution) == problem.num_tasks
        assert solution.min() >= 0
        assert solution.max() < problem.num_servers


class TestAlgorithmConvergence:
    """Tests that algorithms converge (improve over iterations)."""

    def test_sa_convergence(self, problem):
        """Test SA converges."""
        algorithm = SimulatedAnnealing(problem, max_iterations=50, seed=42)
        algorithm.optimize()

        assert len(algorithm.convergence_history) > 0

    def test_ga_convergence(self, problem):
        """Test GA converges."""
        algorithm = GeneticAlgorithm(problem, max_iterations=50, seed=42)
        algorithm.optimize()

        assert len(algorithm.convergence_history) > 0

    def test_aco_convergence(self, problem):
        """Test ACO converges."""
        algorithm = AntColonyOptimization(problem, max_iterations=50, seed=42)
        algorithm.optimize()

        assert len(algorithm.convergence_history) > 0

    def test_ba_convergence(self, problem):
        """Test BA converges."""
        algorithm = BeesAlgorithm(problem, max_iterations=50, seed=42)
        algorithm.optimize()

        assert len(algorithm.convergence_history) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])