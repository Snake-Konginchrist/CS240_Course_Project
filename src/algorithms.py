"""Graph partitioning algorithms."""

from __future__ import annotations

import itertools
from collections import defaultdict

import networkx as nx
import numpy as np

from metrics import Partition, cut_weight


def greedy_balanced_partition(graph: nx.Graph, k: int, seed: int = 0) -> Partition:
    """Assign nodes greedily while maintaining balanced partition sizes."""
    rng = np.random.default_rng(seed)
    nodes = list(graph.nodes())
    rng.shuffle(nodes)
    max_size = int(np.ceil(len(nodes) / k))
    sizes = [0] * k
    partition: Partition = {}

    degrees = dict(graph.degree(weight="weight"))
    nodes.sort(key=lambda node: degrees[node], reverse=True)
    for node in nodes:
        feasible = [part for part in range(k) if sizes[part] < max_size]
        best_part = min(
            feasible,
            key=lambda part: (_incremental_cut_cost(graph, node, part, partition), sizes[part]),
        )
        partition[node] = best_part
        sizes[best_part] += 1
    return partition


def spectral_partition(graph: nx.Graph, k: int) -> Partition:
    """Recursively partition a graph using spectral bisection."""
    groups = [set(graph.nodes())]
    while len(groups) < k:
        index = max(range(len(groups)), key=lambda idx: len(groups[idx]))
        group = groups.pop(index)
        left, right = _spectral_bisect(graph.subgraph(group).copy())
        groups.extend([left, right])

    partition: Partition = {}
    for part, nodes in enumerate(groups):
        for node in nodes:
            partition[node] = part
    return partition


def kernighan_lin_partition(
    graph: nx.Graph,
    k: int,
    initial: Partition | None = None,
    max_passes: int = 8,
    seed: int = 0,
) -> Partition:
    """Refine a k-way partition using pairwise Kernighan-Lin passes."""
    partition = dict(initial or spectral_partition(graph, k))
    for _ in range(max_passes):
        before = cut_weight(graph, partition)
        for a, b in itertools.combinations(range(k), 2):
            partition = _refine_pair(graph, partition, a, b)
        after = cut_weight(graph, partition)
        if after >= before - 1e-9:
            break
    return _renumber_parts(partition)


def _incremental_cut_cost(
    graph: nx.Graph, node: int, part: int, partition: Partition
) -> float:
    cost = 0.0
    for neighbor, edge_data in graph[node].items():
        if neighbor in partition and partition[neighbor] != part:
            cost += float(edge_data.get("weight", 1.0))
    return cost


def _spectral_bisect(graph: nx.Graph) -> tuple[set[int], set[int]]:
    nodes = list(graph.nodes())
    if len(nodes) <= 1:
        return set(nodes), set()

    adjacency = nx.to_numpy_array(graph, nodelist=nodes, weight="weight", dtype=float)
    degrees = np.diag(adjacency.sum(axis=1))
    laplacian = degrees - adjacency
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    fiedler = eigenvectors[:, 1] if len(nodes) > 1 else eigenvectors[:, 0]
    order = np.argsort(fiedler)
    midpoint = len(nodes) // 2
    left = {nodes[i] for i in order[:midpoint]}
    right = {nodes[i] for i in order[midpoint:]}
    return left, right


def _refine_pair(graph: nx.Graph, partition: Partition, a: int, b: int) -> Partition:
    nodes = [node for node, part in partition.items() if part in {a, b}]
    if len(nodes) < 2:
        return partition

    subgraph = graph.subgraph(nodes).copy()
    set_a = {node for node in nodes if partition[node] == a}
    set_b = {node for node in nodes if partition[node] == b}
    if not set_a or not set_b:
        return partition

    try:
        new_a, new_b = nx.algorithms.community.kernighan_lin_bisection(
            subgraph, partition=(set_a, set_b), weight="weight", max_iter=10
        )
    except nx.NetworkXError:
        return partition

    updated = dict(partition)
    for node in new_a:
        updated[node] = a
    for node in new_b:
        updated[node] = b
    if cut_weight(graph, updated) <= cut_weight(graph, partition):
        return updated
    return partition


def _renumber_parts(partition: Partition) -> Partition:
    mapping = {part: idx for idx, part in enumerate(sorted(set(partition.values())))}
    return {node: mapping[part] for node, part in partition.items()}
