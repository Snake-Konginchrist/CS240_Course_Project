"""Basic checks for graph partitioning metrics."""

import networkx as nx

from metrics import balance_ratio, cut_weight


def test_cut_weight_counts_cross_edges_only() -> None:
    graph = nx.path_graph(4)
    partition = {0: 0, 1: 0, 2: 1, 3: 1}
    assert cut_weight(graph, partition) == 1.0


def test_balance_ratio_is_one_for_equal_parts() -> None:
    graph = nx.path_graph(4)
    partition = {0: 0, 1: 0, 2: 1, 3: 1}
    assert balance_ratio(graph, partition, 2) == 1.0
