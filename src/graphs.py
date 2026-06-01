"""Graph generation utilities for influence maximization experiments."""

from __future__ import annotations

import networkx as nx
import numpy as np


def make_erdos_renyi_graph(n_nodes: int, probability: float, seed: int) -> nx.DiGraph:
    """Create a directed Erdos-Renyi graph with IC probabilities."""
    undirected = nx.gnp_random_graph(n_nodes, probability, seed=seed)
    graph = _make_bidirected(undirected)
    set_weighted_cascade_probabilities(graph)
    return graph


def make_barabasi_albert_graph(n_nodes: int, attachment: int, seed: int) -> nx.DiGraph:
    """Create a directed Barabasi-Albert scale-free benchmark graph."""
    undirected = nx.barabasi_albert_graph(n_nodes, attachment, seed=seed)
    graph = _make_bidirected(undirected)
    set_weighted_cascade_probabilities(graph)
    return graph


def make_watts_strogatz_graph(
    n_nodes: int, nearest_neighbors: int, rewire_probability: float, seed: int
) -> nx.DiGraph:
    """Create a directed Watts-Strogatz small-world benchmark graph."""
    undirected = nx.watts_strogatz_graph(
        n_nodes, nearest_neighbors, rewire_probability, seed=seed
    )
    graph = _make_bidirected(undirected)
    set_weighted_cascade_probabilities(graph)
    return graph


def load_karate_graph() -> nx.DiGraph:
    """Load Zachary's karate club graph as a small real-world benchmark."""
    graph = _make_bidirected(nx.karate_club_graph())
    set_weighted_cascade_probabilities(graph)
    return graph


def make_benchmark_graph(kind: str, size: int, seed: int) -> nx.DiGraph:
    """Build a proposal-aligned influence maximization benchmark graph."""
    if kind == "erdos_renyi":
        return make_erdos_renyi_graph(size, probability=min(0.08, 5 / size), seed=seed)
    if kind == "barabasi_albert":
        return make_barabasi_albert_graph(
            size, attachment=max(1, min(4, size // 20)), seed=seed
        )
    if kind == "watts_strogatz":
        return make_watts_strogatz_graph(
            size,
            nearest_neighbors=min(6, max(2, size // 10)),
            rewire_probability=0.15,
            seed=seed,
        )
    if kind == "karate":
        return load_karate_graph()
    raise ValueError(f"Unknown graph kind: {kind}")


def set_fixed_probabilities(graph: nx.DiGraph, probability: float) -> None:
    """Set every directed edge to the same IC propagation probability."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1].")
    for _, _, data in graph.edges(data=True):
        data["prob"] = float(probability)


def set_weighted_cascade_probabilities(graph: nx.DiGraph) -> None:
    """Set p_uv = 1 / indegree(v), a common IC weighted-cascade rule."""
    indegrees = dict(graph.in_degree())
    for _, target, data in graph.edges(data=True):
        # Incoming probabilities shrink for high-indegree targets, matching the
        # weighted-cascade convention used in many IM experiments.
        data["prob"] = 1.0 / max(1, indegrees[target])


def _make_bidirected(graph: nx.Graph) -> nx.DiGraph:
    directed = nx.DiGraph()
    directed.add_nodes_from(graph.nodes())
    for u, v in graph.edges():
        # Synthetic NetworkX graphs are undirected by default; IC propagation is
        # implemented on directed edges, so we add both directions.
        directed.add_edge(u, v)
        directed.add_edge(v, u)
    return directed
