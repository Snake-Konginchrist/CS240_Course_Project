"""Experiment runner for distributed influence maximization."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from baselines import (
    degree_selection,
    evaluate_fixed_seed_set,
    pagerank_selection,
    random_selection,
)
from distributed import distributed_greedy_influence_maximization
from graphs import make_benchmark_graph
from greedy import celf_influence_maximization, greedy_influence_maximization
from metrics import quality_ratio
from plotting import save_metric_plots
from simulation import estimate_spread


def run_experiments(
    output_dir: Path,
    seed: int = 7,
    simulations: int = 40,
) -> pd.DataFrame:
    """Run proposal-aligned influence maximization benchmarks."""
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    cases = [
        ("erdos_renyi", 40, 5),
        ("erdos_renyi", 70, 6),
        ("barabasi_albert", 40, 5),
        ("barabasi_albert", 70, 6),
        ("watts_strogatz", 60, 6),
        ("karate", 34, 5),
    ]

    for case_id, (graph_kind, size, k) in enumerate(cases):
        graph = make_benchmark_graph(graph_kind, size=size, seed=seed + case_id)
        centralized = greedy_influence_maximization(
            graph, k=k, simulations=simulations, seed=seed + 100 * case_id
        )
        baseline_seed = seed + 1000 * case_id
        results = {
            "random": evaluate_fixed_seed_set(
                graph,
                random_selection(graph, k, seed=baseline_seed),
                simulations=simulations,
                seed=baseline_seed,
            ),
            "degree": evaluate_fixed_seed_set(
                graph,
                degree_selection(graph, k),
                simulations=simulations,
                seed=baseline_seed + 1,
            ),
            "pagerank": evaluate_fixed_seed_set(
                graph,
                pagerank_selection(graph, k),
                simulations=simulations,
                seed=baseline_seed + 2,
            ),
            "greedy": centralized,
            "celf": celf_influence_maximization(
                graph, k=k, simulations=simulations, seed=baseline_seed + 3
            ),
        }

        for n_partitions in [2, 4, 8]:
            for multiplier in [1, 2, 5]:
                local_budget = min(graph.number_of_nodes(), multiplier * k)
                name = f"distributed_m{n_partitions}_q{multiplier}k"
                results[name] = distributed_greedy_influence_maximization(
                    graph,
                    k=k,
                    n_partitions=n_partitions,
                    local_budget=local_budget,
                    simulations=max(10, simulations // 3),
                    seed=baseline_seed + n_partitions * 10 + multiplier,
                )

        evaluation_seed = seed + 50_000 + case_id
        fair_spreads = {
            name: estimate_spread(
                graph, result.seeds, simulations=simulations, seed=evaluation_seed
            )
            for name, result in results.items()
        }
        centralized_fair_spread = fair_spreads["greedy"]

        for algorithm_name, result in results.items():
            records.append(
                {
                    "graph": graph_kind,
                    "size_parameter": size,
                    "n_nodes": graph.number_of_nodes(),
                    "n_edges": graph.number_of_edges(),
                    "k": k,
                    "algorithm": algorithm_name,
                    "seeds": " ".join(map(str, result.seeds)),
                    "selection_estimated_spread": result.estimated_spread,
                    "estimated_spread": fair_spreads[algorithm_name],
                    "quality_ratio_to_greedy": quality_ratio(
                        fair_spreads[algorithm_name], centralized_fair_spread
                    ),
                    "runtime_seconds": result.runtime_seconds,
                    "spread_evaluations": result.spread_evaluations,
                    "transmitted_candidates": result.transmitted_candidates,
                }
            )

    data = pd.DataFrame.from_records(records)
    results_csv = output_dir / "influence_results.csv"
    data.to_csv(results_csv, index=False)
    save_metric_plots(results_csv, output_dir / "figures")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/influence_maximization")
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--simulations", type=int, default=40)
    args = parser.parse_args()
    data = run_experiments(args.output_dir, seed=args.seed, simulations=args.simulations)
    print(data.to_string(index=False))


if __name__ == "__main__":
    main()
