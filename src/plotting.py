"""Plotting helpers for influence maximization experiment outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_metric_plots(results_csv: Path, output_dir: Path) -> None:
    """Save one plot per major project metric."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(results_csv)
    for metric, ylabel in [
        ("estimated_spread", "Estimated IC spread"),
        ("quality_ratio_to_greedy", "Quality ratio to greedy"),
        ("runtime_seconds", "Runtime (seconds)"),
        ("spread_evaluations", "Spread evaluations"),
        ("transmitted_candidates", "Transmitted candidates"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for algorithm, frame in data.groupby("algorithm"):
            grouped = frame.groupby("n_nodes")[metric].mean().sort_index()
            ax.plot(grouped.index, grouped.values, marker="o", label=algorithm)
        ax.set_xlabel("Number of nodes")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(output_dir / f"{metric}.png", dpi=180)
        plt.close(fig)
