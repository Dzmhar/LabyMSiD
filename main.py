"""
Main experiment runner for comparing metaheuristic algorithms on Load Balancing problem.
Runs multiple scenarios (small, medium, large, different data scatters) 10 times each.
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
MAX_ITERATIONS = 100
RESULTS_DIR = Path(__file__).parent / "results"

# --- DEFINICJA SCENARIUSZY BADAWCZYCH ---
SCENARIOS = {
    "1_male_dane": {
        "num_tasks": 20,
        "num_servers": 3,
        "scatter": "normal"
    },
    "2_srednie_dane": {
        "num_tasks": 50,
        "num_servers": 5,
        "scatter": "normal"
    },
    "3_duze_dane": {
        "num_tasks": 200,
        "num_servers": 15,
        "scatter": "normal"
    },
    "4_male_dane_brak_rozrzutu": {
        "num_tasks": 20,
        "num_servers": 3,
        "scatter": "zero"  # Wszystkie zadania trwają tyle samo
    },
    "5_srednie_dane_brak_rozrzutu": {
        "num_tasks": 50,
        "num_servers": 5,
        "scatter": "zero"
    },
    "6_duze_dane_brak_rozrzutu": {
        "num_tasks": 200,
        "num_servers": 15,
        "scatter": "zero"
    },
    "7_male_dane_duzy_rozrzut": {
        "num_tasks": 20,
        "num_servers": 3,
        "scatter": "extreme"  # Wszystkie zadania trwają tyle samo
    },
    "8_srednie_dane_duzy_rozrzutu": {
        "num_tasks": 50,
        "num_servers": 5,
        "scatter": "extreme"
    },
    "9_duze_dane_duzy_rozrzutu": {
        "num_tasks": 200,
        "num_servers": 15,
        "scatter": "extreme"
    }
}


def generate_custom_problem(num_tasks: int, num_servers: int, scatter_type: str,
                            scenario_seed: int) -> LoadBalancingProblem:
    """Generuje problem równoważenia obciążenia w zależności od oczekiwanego rozrzutu danych."""

    # Najpierw wywołujemy standardowe generowanie (ono ustawi w środku seed 42)
    problem = LoadBalancingProblem.generate_test_data(num_tasks=num_tasks, num_servers=num_servers)

    # Teraz ustawiamy unikalny seed dla tego konkretnego scenariusza, aby "nadpisać" to sztywne 42
    np.random.seed(scenario_seed)

    if scatter_type == "normal":
        # Skoro metoda statyczna zawsze dawała to samo przez seed(42),
        # tutaj losujemy nowe czasy zadań, używając poprawnego, unikalnego seeda scenariusza.
        problem.task_times = np.random.randint(10, 101, size=num_tasks, dtype=np.int64)

    elif scatter_type == "zero":
        # Wszystkie zadania zajmują dokładnie 10 jednostek czasu.
        # Nadpisujemy właściwą zmienną wewnątrz obiektu: 'task_times' zamiast 'processing_times'
        problem.task_times = np.full(num_tasks, 10, dtype=np.int64)

    elif scatter_type == "extreme":
        # Skrajny rozrzut: od bardzo małych do bardzo dużych liczb całkowitych (np. od 5 do 5000)
        # Musimy zaokrąglić i rzutować na int64, bo Twój problem operuje na takich liczbach.
        float_times = 10 ** np.random.uniform(0.7, 3.7, size=num_tasks)
        problem.task_times = np.clip(float_times, 5, 5000).astype(np.int64)

    return problem


def run_algorithm(
    algorithm_class,
    problem: LoadBalancingProblem,
    seed: int,
    max_iterations: int = MAX_ITERATIONS,
) -> Tuple[np.ndarray, float, float, List[float]]:
    """Run a single algorithm and return results."""
    algorithm = algorithm_class(problem, max_iterations=max_iterations, seed=seed)
    solution, score, exec_time = algorithm.run_with_timing()
    return solution, score, exec_time, algorithm.convergence_history


def run_experiments_for_scenario(scenario_name: str, config: dict) -> Tuple[pd.DataFrame, dict]:
    """Uruchamia komplet testów dla pojedynczego scenariusza danych."""
    # Tworzymy unikalny seed na podstawie nazwy scenariusza (np. "1_male_dane" da inny seed niż "7_male_dane_duzy_rozrzut")
    scenario_seed = abs(hash(scenario_name)) % 10000

    # Generowanie danych wejściowych z uwzględnieniem poprawek zmiennych i seedowania
    problem = generate_custom_problem(
        num_tasks=config["num_tasks"],
        num_servers=config["num_servers"],
        scatter_type=config["scatter"],
        scenario_seed=scenario_seed
    )

    # Dopiero TUTAJ ustawiamy stały punkt startowy dla powtórzeń algorytmów (żeby porównanie algorytmów było sprawiedliwe)
    np.random.seed(42)
    random_seeds = np.random.randint(0, 10000, size=NUM_RUNS)

    algorithms = [
        (SimulatedAnnealing, "Simulated Annealing"),
        (GeneticAlgorithm, "Genetic Algorithm"),
        (AntColonyOptimization, "Ant Colony Optimization"),
        (BeesAlgorithm, "Bees Algorithm"),
    ]

    results = []
    convergence_data = {}

    for algorithm_class, name in algorithms:
        print(f"  Uruchamianie {name}...")
        scores = []
        times = []
        best_histories = []

        for run_idx in range(NUM_RUNS):
            seed = random_seeds[run_idx]
            solution, score, exec_time, history = run_algorithm(algorithm_class, problem, seed)

            scores.append(score)
            times.append(exec_time)
            best_histories.append(history)

        avg_score = np.mean(scores)
        best_score = np.min(scores)
        avg_time = np.mean(times)

        results.append({
            "algorytm": name,
            "najlepszy wynik": best_score,
            "średni wynik": avg_score,
            "czas": avg_time,
        })

        avg_history = [np.mean([h[i] for h in best_histories]) for i in range(len(best_histories[0]))]
        convergence_data[name] = avg_history

    df = pd.DataFrame(results)
    return df, convergence_data


def plot_convergence(convergence_data: dict, output_path: Path, title_suffix: str) -> None:
    """Generate convergence plot."""
    plt.figure(figsize=(10, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for (name, history), color in zip(convergence_data.items(), colors):
        x = range(len(history))
        plt.plot(x, history, label=name, color=color, linewidth=2)

    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Makespan (objective value)", fontsize=12)
    plt.title(f"Convergence of Algorithms ({title_suffix})", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_time_comparison(df: pd.DataFrame, output_path: Path, title_suffix: str) -> None:
    """Generate time comparison bar chart."""
    plt.figure(figsize=(10, 6))
    algorithms = df["algorytm"]
    times = df["czas"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    plt.bar(algorithms, times, color=colors, edgecolor="black")
    plt.xlabel("Algorithm", fontsize=12)
    plt.ylabel("Average Execution Time (seconds)", fontsize=12)
    plt.title(f"Execution Time Comparison ({title_suffix})", fontsize=14)

    for i, t in enumerate(times):
        plt.text(i, t + (max(times) * 0.01), f"{t:.4f}s", ha="center", fontsize=10)

    plt.ylim(0, max(times) * 1.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    """Główna pętla uruchamiająca wszystkie testy po kolei."""
    RESULTS_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("ROZPOCZĘCIE AUTOMATYCZNYCH TESTÓW PORÓWNAWCZYCH")
    print("=" * 60)

    for scenario_id, config in SCENARIOS.items():
        print(f"\n>>> TEST: {scenario_id.upper().replace('_', ' ')}")
        print(f"    Konfiguracja: Zadania={config['num_tasks']}, Serwery={config['num_servers']}, Rozrzut={config['scatter']}\n")

        # Tworzenie dedykowanego podfolderu dla danego scenariusza
        scenario_dir = RESULTS_DIR / scenario_id
        scenario_dir.mkdir(exist_ok=True)

        # Ścieżki zapisu plików dla bieżącego testu
        csv_path = scenario_dir / "wyniki.csv"
        convergence_plot_path = scenario_dir / "wykres_zbieznosci.jpg"
        time_plot_path = scenario_dir / "porownanie_czasow.jpg"

        # Uruchomienie obliczeń
        df, convergence_data = run_experiments_for_scenario(scenario_id, config)

        # Zapis wyników tekstowych (.csv)
        df.to_csv(csv_path, index=False)
        print(f"  [Zapisano] Tabele wyników: {csv_path.name}")

        # Generowanie i zapis wykresów (.jpg)
        friendly_title = scenario_id.split("_", 1)[1].replace("_", " ")

        plot_convergence(convergence_data, convergence_plot_path, friendly_title)
        print(f"  [Zapisano] Wykres zbieżności: {convergence_plot_path.name}")

        plot_time_comparison(df, time_plot_path, friendly_title)
        print(f"  [Zapisano] Wykres czasów: {time_plot_path.name}")

    print("\n" + "=" * 60)
    print("WSZYSTKIE TESTY ZAKOŃCZONE POMYŚLNIE!")
    print(f"Wyniki znajdziesz w folderze: {RESULTS_DIR.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()