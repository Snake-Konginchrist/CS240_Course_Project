"""Basic checks for influence maximization metrics."""

import networkx as nx

from metrics import communication_candidates, marginal_gain, quality_ratio


def test_quality_ratio_handles_zero_reference() -> None:
    assert quality_ratio(3.0, 0.0) == 0.0
    assert quality_ratio(3.0, 6.0) == 0.5


def test_communication_candidates_is_m_times_q() -> None:
    assert communication_candidates(4, 10) == 40


def test_marginal_gain_on_deterministic_edge() -> None:
    graph = nx.DiGraph()
    graph.add_edge(0, 1, prob=1.0)
    assert marginal_gain(graph, [], 0, simulations=3, seed=1) == 2.0
