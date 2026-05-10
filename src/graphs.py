"""Graph generation utilities."""

from __future__ import annotations

import networkx as nx
import numpy as np


def make_grid_graph(side: int) -> nx.Graph:
    """Create a 2D grid graph with integer node labels."""
    graph = nx.grid_2d_graph(side, side)
    graph = nx.convert_node_labels_to_integers(graph)
    _set_unit_weights(graph)
    return graph


def make_erdos_renyi_graph(n_nodes: int, probability: float, seed: int) -> nx.Graph:
    """Create a connected Erdos-Renyi graph."""
    rng = np.random.default_rng(seed)
    for attempt in range(100):
        graph = nx.gnp_random_graph(n_nodes, probability, seed=int(rng.integers(1_000_000)))
        if nx.is_connected(graph):
            _set_unit_weights(graph)
            return graph
    raise RuntimeError("Failed to generate a connected Erdos-Renyi graph.")


def make_random_geometric_graph(n_nodes: int, radius: float, seed: int) -> nx.Graph:
    """Create a connected random geometric graph."""
    rng = np.random.default_rng(seed)
    for attempt in range(100):
        graph = nx.random_geometric_graph(n_nodes, radius, seed=int(rng.integers(1_000_000)))
        graph = nx.convert_node_labels_to_integers(graph)
        if nx.is_connected(graph):
            _set_unit_weights(graph)
            return graph
    raise RuntimeError("Failed to generate a connected random geometric graph.")


def load_karate_graph() -> nx.Graph:
    """Load the Zachary karate club graph as a small real-world benchmark."""
    graph = nx.karate_club_graph()
    _set_unit_weights(graph)
    return graph


def make_benchmark_graph(kind: str, size: int, seed: int) -> nx.Graph:
    """Build a benchmark graph by name."""
    if kind == "grid":
        return make_grid_graph(size)
    if kind == "erdos_renyi":
        return make_erdos_renyi_graph(size, probability=min(0.12, 6 / size), seed=seed)
    if kind == "geometric":
        return make_random_geometric_graph(size, radius=min(0.35, 2.5 / np.sqrt(size)), seed=seed)
    if kind == "karate":
        return load_karate_graph()
    raise ValueError(f"Unknown graph kind: {kind}")


def _set_unit_weights(graph: nx.Graph) -> None:
    for _, _, data in graph.edges(data=True):
        data["weight"] = float(data.get("weight", 1.0))

