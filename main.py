"""
Main experiment runner for comparing metaheuristic algorithms on Load Balancing problem.

This script runs each algorithm 10 times and collects statistics:
- Best score
- Average score
- Execution time

Results are saved to results.csv and convergence/time charts are generated.
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.problem import LoadBalancingProblem
from core.algorithms.simulated_annealing import SimulatedAnnealing
from core.algorithms.genetic_algorithm import GeneticAlgorithm
from core.algorithms.ant_colony import AntColonyOptimization
from core.algorithms.bees_algorithm import BeesAlgorithm


NUM_RUNS = 10
NUM_TASKS = 50
NUM_SERVERS = 5
MAX_ITERATIONS = 100

RESULTS_DIR = Path(__file__).parent / "results"
CSV_PATH = RESULTS_DIR / "results.csv"
CONVERGENCE_PLOT_PATH = RESULTS_DIR / "convergence_plot.png"
TIME_PLOT_PATH = RESULTS_DIR / "time_comparison.png"


def run_algorithm(
    algorithm_class,
    problem: LoadBalancingProblem,
    seed: int,
    max_iterations: int = MAX_ITERATIONS,
) -> Tuple[np.ndarray, float, float, List[float]]:
    """
    Run a single algorithm and return results.

    Args:
        algorithm_class: The algorithm class to instantiate.
        problem: The problem instance.
        seed: Random seed.
        max_iterations: Max iterations for the algorithm.

    Returns:
        Tuple of (solution, score, execution_time, convergence_history).
    """
    algorithm = algorithm_class(problem, max_iterations=max_iterations, seed=seed)

    solution, score, exec_time = algorithm.run_with_timing()

    return solution, score, exec_time, algorithm.convergence_history


def run_experiments() -> pd.DataFrame:
    """
    Run experiments for all algorithms.

    Returns:
        DataFrame with results.
    """
    np.random.seed(42)
    random_seeds = np.random.randint(0, 10000, size=NUM_RUNS)

    problem = LoadBalancingProblem.generate_test_data(
        num_tasks=NUM_TASKS, num_servers=NUM_SERVERS
    )

    algorithms = [
        (SimulatedAnnealing, "Simulated Annealing"),
        (GeneticAlgorithm, "Genetic Algorithm"),
        (AntColonyOptimization, "Ant Colony Optimization"),
        (BeesAlgorithm, "Bees Algorithm"),
    ]

    results = []
    convergence_data = {}

    for algorithm_class, name in algorithms:
        print(f"Running {name}...")

        scores = []
        times = []
        best_histories = []

        for run_idx in range(NUM_RUNS):
            seed = random_seeds[run_idx]

            solution, score, exec_time, history = run_algorithm(
                algorithm_class, problem, seed
            )

            scores.append(score)
            times.append(exec_time)
            best_histories.append(history)

            print(f"  Run {run_idx + 1}: score={score:.2f}, time={exec_time:.4f}s")

        avg_score = np.mean(scores)
        best_score = np.min(scores)
        avg_time = np.mean(times)

        print(f"  {name}: best={best_score:.2f}, avg={avg_score:.2f}, time={avg_time:.4f}s\n")

        results.append(
            {
                "algorytm": name,
                "najlepszy wynik": best_score,
                "średni wynik": avg_score,
                "czas": avg_time,
            }
        )

        avg_history = [np.mean([h[i] for h in best_histories]) for i in range(len(best_histories[0]))]
        convergence_data[name] = avg_history

    df = pd.DataFrame(results)
    return df, convergence_data


def plot_convergence(convergence_data: dict, output_path: Path) -> None:
    """
    Generate convergence plot.

    Args:
        convergence_data: Dictionary mapping algorithm name to convergence history.
        output_path: Path to save the plot.
    """
    plt.figure(figsize=(10, 6))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for (name, history), color in zip(convergence_data.items(), colors):
        x = range(len(history))
        plt.plot(x, history, label=name, color=color, linewidth=2)

    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Makespan (objective value)", fontsize=12)
    plt.title("Convergence of Metaheuristic Algorithms", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved convergence plot to {output_path}")


def plot_time_comparison(df: pd.DataFrame, output_path: Path) -> None:
    """
    Generate time comparison bar chart.

    Args:
        df: Results DataFrame.
        output_path: Path to save the plot.
    """
    plt.figure(figsize=(10, 6))

    algorithms = df["algorytm"]
    times = df["czas"]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    plt.bar(algorithms, times, color=colors, edgecolor="black")

    plt.xlabel("Algorithm", fontsize=12)
    plt.ylabel("Average Execution Time (seconds)", fontsize=12)
    plt.title("Execution Time Comparison", fontsize=14)

    for i, t in enumerate(times):
        plt.text(i, t + 0.01, f"{t:.4f}s", ha="center", fontsize=10)

    plt.ylim(0, max(times) * 1.2)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved time comparison plot to {output_path}")


def main():
    """Main entry point."""
    RESULTS_DIR.mkdir(exist_ok=True)

    print(f"Running experiments with {NUM_RUNS} runs each...")
    print(f"Problem: {NUM_TASKS} tasks, {NUM_SERVERS} servers, {MAX_ITERATIONS} iterations\n")

    df, convergence_data = run_experiments()

    df.to_csv(CSV_PATH, index=False)
    print(f"\nSaved results to {CSV_PATH}")

    plot_convergence(convergence_data, CONVERGENCE_PLOT_PATH)
    plot_time_comparison(df, TIME_PLOT_PATH)

    print("\nDone!")


if __name__ == "__main__":
    main()