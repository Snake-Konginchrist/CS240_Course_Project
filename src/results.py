"""Shared result types for influence maximization algorithms."""

from __future__ import annotations

from dataclasses import dataclass


SeedSet = list[int]


@dataclass(frozen=True)
class SelectionResult:
    """Common result object returned by seed-selection algorithms."""

    seeds: SeedSet
    estimated_spread: float
    spread_evaluations: int
    runtime_seconds: float
    transmitted_candidates: int = 0
