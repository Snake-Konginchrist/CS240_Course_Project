"""Plotting helpers for experiment outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def save_metric_plots(results_csv: Path, output_dir: Path) -> None:
    """Save one plot per major metric."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(results_csv)
    for metric, ylabel in [
        ("cut_weight", "Cut weight"),
        ("runtime_seconds", "Runtime (seconds)"),
        ("balance_ratio", "Balance ratio"),
        ("consensus_error", "Consensus error"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4))
        for algorithm, frame in data.groupby("algorithm"):
            grouped = frame.groupby("n_nodes")[metric].mean().sort_index()
            ax.plot(grouped.index, grouped.values, marker="o", label=algorithm)
        ax.set_xlabel("Number of nodes")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"{metric}.png", dpi=180)
        plt.close(fig)

