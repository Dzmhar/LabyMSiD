"""
Problem definition for the Load Balancing (Task Scheduling) problem.

This module defines the structure for representing the load balancing problem,
including task processing times, server assignments, and the makespan objective function.
"""

import numpy as np
from typing import Tuple


class LoadBalancingProblem:
    """
    Represents the Load Balancing (Task Scheduling) problem.

    The goal is to assign N tasks (each with a specific processing time) to M servers
    to minimize the makespan (the completion time of the most loaded server).
    """

    def __init__(self, num_tasks: int, num_servers: int, task_times: np.ndarray | None = None):
        """
        Initialize the load balancing problem.

        Args:
            num_tasks: Number of tasks to schedule.
            num_servers: Number of available servers.
            task_times: Optional array of task processing times. If None, random times
                       will be generated between 10 and 100.
        """
        self.num_tasks = num_tasks
        self.num_servers = num_servers

        if task_times is not None:
            if len(task_times) != num_tasks:
                raise ValueError(f"task_times length {len(task_times)} must equal num_tasks {num_tasks}")
            self.task_times = np.array(task_times, dtype=np.int64)
        else:
            self.task_times = np.random.randint(10, 101, size=num_tasks, dtype=np.int64)

    def calculate_makespan(self, solution: np.ndarray) -> float:
        """
        Calculate the makespan for a given solution.

        The makespan is the maximum sum of task processing times across all servers.

        Args:
            solution: A numpy array where index represents Task ID and value represents Server ID.

        Returns:
            The makespan (maximum load across all servers).
        """
        solution = np.asarray(solution)
        server_loads = np.zeros(self.num_servers, dtype=np.float64)

        for task_id, server_id in enumerate(solution):
            server_loads[server_id] += self.task_times[task_id]

        return float(np.max(server_loads))

    def calculate_server_loads(self, solution: np.ndarray) -> np.ndarray:
        """
        Calculate the load for each server given a solution.

        Args:
            solution: A numpy array where index represents Task ID and value represents Server ID.

        Returns:
            An array of server loads.
        """
        solution = np.asarray(solution)
        server_loads = np.zeros(self.num_servers, dtype=np.float64)

        for task_id, server_id in enumerate(solution):
            server_loads[server_id] += self.task_times[task_id]

        return server_loads

    def generate_random_solution(self) -> np.ndarray:
        """
        Generate a random valid solution.

        Returns:
            A numpy array where each element is a random server ID between 0 and num_servers-1.
        """
        return np.random.randint(0, self.num_servers, size=self.num_tasks, dtype=np.int64)

    def validate_solution(self, solution: np.ndarray) -> Tuple[bool, str]:
        """
        Validate a solution.

        Args:
            solution: The solution array to validate.

        Returns:
            A tuple (is_valid, message).
        """
        solution = np.asarray(solution)

        if len(solution) != self.num_tasks:
            return False, f"Solution length {len(solution)} does not match num_tasks {self.num_tasks}"

        if solution.min() < 0 or solution.max() >= self.num_servers:
            return False, f"Server IDs must be between 0 and {self.num_servers - 1}"

        return True, "Valid"

    @staticmethod
    def generate_test_data(num_tasks: int = 50, num_servers: int = 5) -> "LoadBalancingProblem":
        """
        Generate a standard test problem instance.

        Args:
            num_tasks: Number of tasks (default 50).
            num_servers: Number of servers (default 5).

        Returns:
            A LoadBalancingProblem instance.
        """
        np.random.seed(42)
        return LoadBalancingProblem(num_tasks=num_tasks, num_servers=num_servers)