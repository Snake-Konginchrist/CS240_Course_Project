"""GreeDI-style distributed influence maximization."""

from __future__ import annotations

import time

import networkx as nx
import numpy as np

from greedy import greedy_influence_maximization
from results import SelectionResult


def distributed_greedy_influence_maximization(
    graph: nx.DiGraph,
    k: int,
    n_partitions: int,
    local_budget: int,
    simulations: int = 200,
    seed: int = 0,
) -> SelectionResult:
    """Run a local-candidate-sharing algorithm inspired by GreeDI.

    Each partition runs local greedy on assigned nodes, sends at most
    ``local_budget`` candidates, and the coordinator runs greedy on their union.
    """
    start = time.perf_counter()
    partitions = random_node_partitions(graph, n_partitions, seed=seed)
    candidate_pool: set[int] = set()
    evaluations = 0

    for idx, local_nodes in enumerate(partitions):
        # Worker phase: each partition sees only its local candidate list, but
        # spread is still evaluated on the full graph to model global diffusion.
        local_result = greedy_influence_maximization(
            graph,
            k=min(local_budget, len(local_nodes)),
            simulations=simulations,
            seed=seed + idx + 1,
            candidates=local_nodes,
        )
        evaluations += local_result.spread_evaluations
        candidate_pool.update(local_result.seeds)

    # Communication proxy used in the report: each worker sends up to q nodes.
    transmitted_candidates = sum(min(local_budget, len(part)) for part in partitions)

    # Coordinator phase: run greedy again, but only over the merged candidate
    # pool rather than over all graph nodes.
    coordinator = greedy_influence_maximization(
        graph,
        k=k,
        simulations=simulations,
        seed=seed + 10_000,
        candidates=sorted(candidate_pool),
    )
    evaluations += coordinator.spread_evaluations

    return SelectionResult(
        seeds=coordinator.seeds,
        estimated_spread=coordinator.estimated_spread,
        spread_evaluations=evaluations,
        runtime_seconds=time.perf_counter() - start,
        transmitted_candidates=transmitted_candidates,
    )


def random_node_partitions(
    graph: nx.DiGraph, n_partitions: int, seed: int = 0
) -> list[list[int]]:
    """Uniformly partition nodes into n_partitions local worker lists."""
    if n_partitions <= 0:
        raise ValueError("n_partitions must be positive.")
    rng = np.random.default_rng(seed)
    nodes = list(graph.nodes())
    rng.shuffle(nodes)

    # np.array_split keeps partition sizes as balanced as possible.
    return [list(chunk) for chunk in np.array_split(nodes, n_partitions)]
