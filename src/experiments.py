"""Experiment runner for graph partitioning benchmarks."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from algorithms import (
    greedy_balanced_partition,
    kernighan_lin_partition,
    spectral_partition,
)
from graphs import make_benchmark_graph
from metrics import balance_ratio, cut_weight, normalized_cut
from plotting import save_metric_plots
from simulation import consensus_averaging


def run_experiments(output_dir: Path, seed: int = 7) -> pd.DataFrame:
    """Run all benchmarks and save raw results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    cases = [
        ("grid", 6, 4),
        ("grid", 8, 4),
        ("grid", 10, 4),
        ("erdos_renyi", 40, 4),
        ("erdos_renyi", 80, 4),
        ("geometric", 50, 4),
        ("geometric", 90, 4),
        ("karate", 34, 2),
    ]
    algorithms = {
        "greedy": lambda graph, k: greedy_balanced_partition(graph, k, seed=seed),
        "spectral": lambda graph, k: spectral_partition(graph, k),
        "kernighan_lin": lambda graph, k: kernighan_lin_partition(
            graph, k, initial=spectral_partition(graph, k), seed=seed
        ),
    }

    for graph_kind, size, k in cases:
        graph = make_benchmark_graph(graph_kind, size=size, seed=seed)
        for algorithm_name, algorithm in algorithms.items():
            start = time.perf_counter()
            partition = algorithm(graph, k)
            runtime = time.perf_counter() - start
            consensus = consensus_averaging(graph, partition, seed=seed)
            records.append(
                {
                    "graph": graph_kind,
                    "size_parameter": size,
                    "n_nodes": graph.number_of_nodes(),
                    "n_edges": graph.number_of_edges(),
                    "k": k,
                    "algorithm": algorithm_name,
                    "cut_weight": cut_weight(graph, partition),
                    "normalized_cut": normalized_cut(graph, partition, k),
                    "balance_ratio": balance_ratio(graph, partition, k),
                    "runtime_seconds": runtime,
                    **consensus,
                }
            )

    data = pd.DataFrame.from_records(records)
    results_csv = output_dir / "partition_results.csv"
    data.to_csv(results_csv, index=False)
    save_metric_plots(results_csv, output_dir / "figures")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    data = run_experiments(args.output_dir, seed=args.seed)
    print(data.to_string(index=False))


if __name__ == "__main__":
    main()
