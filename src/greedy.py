"""Centralized greedy and CELF algorithms."""

from __future__ import annotations

import heapq
import time
from collections.abc import Iterable

import networkx as nx
import numpy as np

from results import SeedSet, SelectionResult
from simulation import estimate_spread


def greedy_influence_maximization(
    graph: nx.DiGraph,
    k: int,
    simulations: int = 200,
    seed: int = 0,
    candidates: Iterable[int] | None = None,
) -> SelectionResult:
    """Run centralized Monte Carlo greedy influence maximization."""
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    candidate_nodes = list(candidates if candidates is not None else graph.nodes())
    selected: SeedSet = []
    evaluations = 0
    current_spread = _estimate(graph, selected, simulations, rng)
    evaluations += 1

    while len(selected) < k and candidate_nodes:
        best_node = None
        best_spread = -1.0
        selected_set = set(selected)

        # Greedy step: try every unselected candidate and keep the node with
        # the largest estimated marginal contribution to the current seed set.
        for node in candidate_nodes:
            if node in selected_set:
                continue
            trial_spread = _estimate(graph, [*selected, node], simulations, rng)
            evaluations += 1
            if trial_spread > best_spread:
                best_node = node
                best_spread = trial_spread
        if best_node is None:
            break
        selected.append(best_node)
        current_spread = best_spread

    return SelectionResult(
        seeds=selected,
        estimated_spread=current_spread,
        spread_evaluations=evaluations,
        runtime_seconds=time.perf_counter() - start,
    )


def celf_influence_maximization(
    graph: nx.DiGraph,
    k: int,
    simulations: int = 200,
    seed: int = 0,
    candidates: Iterable[int] | None = None,
) -> SelectionResult:
    """Run CELF/lazy greedy influence maximization.

    Cached marginal gains are valid upper bounds after the seed set grows,
    because the IC influence objective is monotone submodular.
    """
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    candidate_nodes = list(candidates if candidates is not None else graph.nodes())
    selected: SeedSet = []
    selected_set: set[int] = set()
    evaluations = 0
    current_spread = _estimate(graph, selected, simulations, rng)
    evaluations += 1

    # The heap stores (-cached_marginal_gain, node, selected_size_when_computed).
    # Python has a min-heap, so negating the gain gives max-priority behavior.
    heap: list[tuple[float, int, int]] = []
    for node in candidate_nodes:
        spread = _estimate(graph, [node], simulations, rng)
        evaluations += 1
        marginal = spread - current_spread
        heapq.heappush(heap, (-marginal, node, 0))

    while len(selected) < k and heap:
        neg_gain, node, last_updated = heapq.heappop(heap)
        if node in selected_set:
            continue

        # If the cached gain was computed for the current seed set, CELF can
        # safely accept this node without recomputing every other candidate.
        if last_updated == len(selected):
            selected.append(node)
            selected_set.add(node)
            current_spread += -neg_gain
            continue

        # Otherwise the cached gain is stale. Recompute this single candidate's
        # marginal gain and push it back; submodularity keeps old gains as upper
        # bounds, which is why lazy evaluation is correct.
        spread = _estimate(graph, [*selected, node], simulations, rng)
        evaluations += 1
        marginal = spread - current_spread
        heapq.heappush(heap, (-marginal, node, len(selected)))

    # Re-evaluate the final seed set because CELF's running current_spread is
    # assembled from marginal estimates that may come from different MC samples.
    final_spread = _estimate(graph, selected, simulations, rng)
    evaluations += 1
    return SelectionResult(
        seeds=selected,
        estimated_spread=final_spread,
        spread_evaluations=evaluations,
        runtime_seconds=time.perf_counter() - start,
    )


def _estimate(
    graph: nx.DiGraph, seeds: Iterable[int], simulations: int, rng: np.random.Generator
) -> float:
    """Draw a fresh deterministic sub-seed for each Monte Carlo estimate."""
    return estimate_spread(
        graph, seeds, simulations=simulations, seed=int(rng.integers(1_000_000_000))
    )
