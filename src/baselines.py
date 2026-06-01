"""Simple non-greedy baselines for influence maximization."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable

import networkx as nx
import numpy as np

from results import SeedSet, SelectionResult
from simulation import estimate_spread


def random_selection(graph: nx.DiGraph, k: int, seed: int = 0) -> SeedSet:
    """Select k uniformly random nodes."""
    rng = np.random.default_rng(seed)
    nodes = list(graph.nodes())
    if k >= len(nodes):
        return nodes
    return list(rng.choice(nodes, size=k, replace=False))


def degree_selection(graph: nx.DiGraph, k: int) -> SeedSet:
    """Select nodes by weighted out-degree."""
    return _top_k(graph.nodes(), k, key=lambda node: graph.out_degree(node, weight="prob"))


def pagerank_selection(graph: nx.DiGraph, k: int) -> SeedSet:
    """Select nodes by PageRank score."""
    if graph.number_of_nodes() == 0:
        return []
    scores = nx.pagerank(graph, weight="prob")
    return _top_k(graph.nodes(), k, key=lambda node: scores[node])


def evaluate_fixed_seed_set(
    graph: nx.DiGraph, seeds: Iterable[int], simulations: int = 500, seed: int = 0
) -> SelectionResult:
    """Evaluate a baseline seed set with the same result shape as greedy methods."""
    start = time.perf_counter()
    spread = estimate_spread(graph, seeds, simulations=simulations, seed=seed)
    return SelectionResult(
        seeds=list(seeds),
        estimated_spread=spread,
        spread_evaluations=1,
        runtime_seconds=time.perf_counter() - start,
    )


def _top_k(nodes: Iterable[int], k: int, key: Callable[[int], float]) -> SeedSet:
    return sorted(nodes, key=lambda node: (-key(node), node))[:k]
