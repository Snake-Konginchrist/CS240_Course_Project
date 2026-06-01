"""Basic checks for influence maximization algorithms."""

import networkx as nx

from baselines import degree_selection
from distributed import distributed_greedy_influence_maximization
from greedy import celf_influence_maximization, greedy_influence_maximization


def _path_graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_edge(0, 1, prob=1.0)
    graph.add_edge(1, 2, prob=1.0)
    graph.add_edge(2, 3, prob=1.0)
    return graph


def test_greedy_finds_start_of_deterministic_path() -> None:
    result = greedy_influence_maximization(_path_graph(), k=1, simulations=5, seed=1)
    assert result.seeds == [0]
    assert result.estimated_spread == 4.0


def test_celf_returns_valid_seed_budget() -> None:
    result = celf_influence_maximization(_path_graph(), k=2, simulations=5, seed=1)
    assert len(result.seeds) == 2
    assert set(result.seeds).issubset({0, 1, 2, 3})


def test_distributed_greedy_reports_communication() -> None:
    result = distributed_greedy_influence_maximization(
        _path_graph(), k=2, n_partitions=2, local_budget=1, simulations=5, seed=1
    )
    assert len(result.seeds) == 2
    assert result.transmitted_candidates == 2


def test_degree_baseline_uses_out_degree() -> None:
    assert degree_selection(_path_graph(), k=1) == [0]
