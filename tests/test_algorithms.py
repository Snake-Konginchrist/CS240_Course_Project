"""Basic checks for partitioning algorithms."""

import networkx as nx

from algorithms import (
    greedy_balanced_partition,
    kernighan_lin_partition,
    spectral_partition,
)


def _assert_valid_partition(partition: dict[int, int], n_nodes: int, k: int) -> None:
    assert set(partition) == set(range(n_nodes))
    assert set(partition.values()).issubset(set(range(k)))


def test_partition_algorithms_return_all_nodes() -> None:
    graph = nx.grid_2d_graph(3, 3)
    graph = nx.convert_node_labels_to_integers(graph)
    k = 3
    for algorithm in [
        greedy_balanced_partition,
        spectral_partition,
        kernighan_lin_partition,
    ]:
        partition = algorithm(graph, k)
        _assert_valid_partition(partition, graph.number_of_nodes(), k)
