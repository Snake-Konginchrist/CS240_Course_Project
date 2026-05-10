"""Partition quality metrics."""

from __future__ import annotations

import networkx as nx


Partition = dict[int, int]


def cut_weight(graph: nx.Graph, partition: Partition) -> float:
    """Return the total weight of edges crossing partitions."""
    total = 0.0
    for u, v, data in graph.edges(data=True):
        if partition[u] != partition[v]:
            total += float(data.get("weight", 1.0))
    return total


def balance_ratio(graph: nx.Graph, partition: Partition, k: int) -> float:
    """Return max partition size divided by ideal balanced size."""
    counts = [0] * k
    for node in graph.nodes:
        counts[partition[node]] += 1
    ideal = graph.number_of_nodes() / k
    return max(counts) / ideal


def partition_sizes(partition: Partition, k: int) -> list[int]:
    """Return partition sizes in partition-id order."""
    sizes = [0] * k
    for part in partition.values():
        sizes[part] += 1
    return sizes


def normalized_cut(graph: nx.Graph, partition: Partition, k: int) -> float:
    """Compute a simple normalized cut objective."""
    boundary = [0.0] * k
    volume = [0.0] * k
    for node in graph.nodes:
        part = partition[node]
        volume[part] += float(graph.degree(node, weight="weight"))
    for u, v, data in graph.edges(data=True):
        weight = float(data.get("weight", 1.0))
        if partition[u] != partition[v]:
            boundary[partition[u]] += weight
            boundary[partition[v]] += weight
    return sum(boundary[i] / volume[i] for i in range(k) if volume[i] > 0)

