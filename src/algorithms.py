"""Compatibility exports for all seed-selection algorithms.

For easier navigation, implementation code lives in:

- ``baselines.py`` for random/degree/PageRank baselines
- ``greedy.py`` for centralized greedy and CELF
- ``distributed.py`` for GreeDI-style candidate sharing
"""

from baselines import (
    degree_selection,
    evaluate_fixed_seed_set,
    pagerank_selection,
    random_selection,
)
from distributed import (
    distributed_greedy_influence_maximization,
    random_node_partitions,
)
from greedy import celf_influence_maximization, greedy_influence_maximization
from results import SeedSet, SelectionResult

__all__ = [
    "SeedSet",
    "SelectionResult",
    "random_selection",
    "degree_selection",
    "pagerank_selection",
    "evaluate_fixed_seed_set",
    "greedy_influence_maximization",
    "celf_influence_maximization",
    "distributed_greedy_influence_maximization",
    "random_node_partitions",
]
