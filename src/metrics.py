"""Evaluation metrics for influence maximization experiments."""

from __future__ import annotations

from collections.abc import Iterable

import networkx as nx

from simulation import estimate_spread


def quality_ratio(distributed_spread: float, centralized_spread: float) -> float:
    """Return distributed quality relative to centralized greedy."""
    if centralized_spread <= 0:
        return 0.0
    return distributed_spread / centralized_spread


def communication_candidates(n_partitions: int, local_budget: int) -> int:
    """Communication proxy: number of local candidates sent to the coordinator."""
    return n_partitions * local_budget


def marginal_gain(
    graph: nx.DiGraph,
    seeds: Iterable[int],
    candidate: int,
    simulations: int = 200,
    seed: int = 0,
) -> float:
    """Estimate the marginal IC spread gain of adding one candidate."""
    base = list(seeds)
    with_candidate = [*base, candidate]
    return estimate_spread(graph, with_candidate, simulations, seed) - estimate_spread(
        graph, base, simulations, seed + 1
    )
