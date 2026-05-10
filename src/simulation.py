"""Simple distributed optimization and consensus simulations."""

from __future__ import annotations

import networkx as nx
import numpy as np

from metrics import Partition


def consensus_averaging(
    graph: nx.Graph,
    partition: Partition,
    steps: int = 80,
    seed: int = 0,
) -> dict[str, float]:
    """Run average consensus and report final error plus cross-partition traffic."""
    rng = np.random.default_rng(seed)
    nodes = list(graph.nodes())
    index = {node: i for i, node in enumerate(nodes)}
    values = rng.normal(size=len(nodes))
    target = float(values.mean())
    degree_max = max(dict(graph.degree()).values())
    alpha = 1.0 / (degree_max + 1.0)
    cross_messages = 0

    for _ in range(steps):
        new_values = values.copy()
        for u, v in graph.edges():
            diff = values[index[v]] - values[index[u]]
            new_values[index[u]] += alpha * diff
            new_values[index[v]] -= alpha * diff
            if partition[u] != partition[v]:
                cross_messages += 2
        values = new_values

    error = float(np.linalg.norm(values - target) / np.sqrt(len(nodes)))
    return {
        "consensus_error": error,
        "cross_messages": float(cross_messages),
    }
