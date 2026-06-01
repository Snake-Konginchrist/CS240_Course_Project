"""Independent Cascade simulation utilities."""

from __future__ import annotations

from collections.abc import Iterable

import networkx as nx
import numpy as np


def independent_cascade(
    graph: nx.DiGraph, seeds: Iterable[int], rng: np.random.Generator
) -> set[int]:
    """Simulate one Independent Cascade diffusion run."""
    active = set(seeds)

    # Frontier contains nodes that became active in the previous round. Under
    # IC, only these newly active nodes get one chance to activate neighbors.
    frontier = list(active)

    while frontier:
        next_frontier: list[int] = []
        for node in frontier:
            for neighbor, edge_data in graph[node].items():
                if neighbor in active:
                    continue
                probability = float(edge_data.get("prob", 0.0))

                # Each directed edge attempts activation independently once.
                if rng.random() <= probability:
                    active.add(neighbor)
                    next_frontier.append(neighbor)
        frontier = next_frontier

    return active


def estimate_spread(
    graph: nx.DiGraph,
    seeds: Iterable[int],
    simulations: int = 200,
    seed: int = 0,
) -> float:
    """Estimate expected IC spread by Monte Carlo simulation."""
    if simulations <= 0:
        raise ValueError("simulations must be positive.")
    rng = np.random.default_rng(seed)
    seed_list = list(seeds)
    total = 0

    # Average the final active-set size across independent diffusion runs.
    for _ in range(simulations):
        total += len(independent_cascade(graph, seed_list, rng))
    return total / simulations
