"""Plotting and visualization utilities for InboxRL report and analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set clean academic plotting style
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 1.8,
    "grid.alpha": 0.4,
})


def plot_training_curves(log_csv: Path, out_dir: Path) -> None:
    """Plot total reward and component curves over training steps."""
    if not log_csv.exists():
        print(f"Log file not found: {log_csv}")
        return

    df = pd.read_csv(log_csv)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Total Reward Curve
    fig, ax = plt.subplots(figsize=(6, 4))
    if "step" in df.columns and "reward" in df.columns:
        ax.plot(df["step"], df["reward"], label="Total Reward $R$", color="#1f77b4")
        if "reward_smooth" in df.columns:
            ax.plot(df["step"], df["reward_smooth"], label="Smoothed", color="#ff7f0e", linestyle="--")
        elif len(df) > 10:
            smooth = df["reward"].rolling(window=10, min_periods=1).mean()
            ax.plot(df["step"], smooth, label="10-step Moving Avg", color="#ff7f0e", linestyle="--")

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Reward")
    ax.set_title("GRPO Policy Optimization: Training Reward")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "reward_vs_step.pdf")
    fig.savefig(out_dir / "reward_vs_step.png", dpi=300)
    plt.close(fig)
    print(f"Saved reward curve to {out_dir / 'reward_vs_step.pdf'}")

    # 2. Reward Components
    fig, ax = plt.subplots(figsize=(6, 4))
    comp_cols = [
        ("r_match", "Match Reward ($r_{match}$)", "#2ca02c"),
        ("r_format", "Format Reward ($r_{format}$)", "#1f77b4"),
        ("r_empty", "Empty Precision ($r_{empty}$)", "#9467bd"),
    ]
    for col, lbl, clr in comp_cols:
        if col in df.columns:
            ax.plot(df["step"], df[col], label=lbl, color=clr)

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Component Score")
    ax.set_title("Reward Decomposition During Training")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "reward_components.pdf")
    fig.savefig(out_dir / "reward_components.png", dpi=300)
    plt.close(fig)
    print(f"Saved components curve to {out_dir / 'reward_components.pdf'}")


def plot_baseline_comparison(
    metrics_dict: dict[str, dict[str, float]],
    out_dir: Path,
) -> None:
    """Plot bar chart comparing baselines vs GRPO trained agent."""
    out_dir.mkdir(parents=True, exist_ok=True)
    models = list(metrics_dict.keys())
    
    f1_scores = [metrics_dict[m].get("event_f1", 0.0) * 100 for m in models]
    date_accs = [metrics_dict[m].get("date_match_ratio", 0.0) * 100 for m in models]
    valid_ratios = [metrics_dict[m].get("valid_json_ratio", 0.0) * 100 for m in models]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7, 4.2))
    rects1 = ax.bar(x - width, f1_scores, width, label="Event F1 (%)", color="#2ca02c")
    rects2 = ax.bar(x, date_accs, width, label="Date Match (%)", color="#1f77b4")
    rects3 = ax.bar(x + width, valid_ratios, width, label="JSON Validity (%)", color="#ff7f0e")

    ax.set_ylabel("Score (%)")
    ax.set_title("Performance Comparison Across Baselines & InboxRL")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend(loc="lower right")
    ax.grid(axis="y")
    ax.set_ylim(0, 105)

    plt.tight_layout()
    fig.savefig(out_dir / "baseline_comparison.pdf")
    fig.savefig(out_dir / "baseline_comparison.png", dpi=300)
    plt.close(fig)
    print(f"Saved baseline comparison to {out_dir / 'baseline_comparison.pdf'}")


def plot_curriculum_breakdown(
    zero_shot_levels: dict[int, float],
    grpo_levels: dict[int, float],
    out_dir: Path,
) -> None:
    """Plot accuracy comparison by difficulty level (L1 to L5)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    levels = sorted(list(set(zero_shot_levels.keys()) | set(grpo_levels.keys())))
    if not levels:
        levels = [1, 2, 3, 4, 5]

    zs_scores = [zero_shot_levels.get(lvl, 0.0) * 100 for lvl in levels]
    grpo_scores = [grpo_levels.get(lvl, 0.0) * 100 for lvl in levels]

    x = np.arange(len(levels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(x - width / 2, zs_scores, width, label="Zero-Shot Base", color="#7f7f7f")
    ax.bar(x + width / 2, grpo_scores, width, label="InboxRL (GRPO)", color="#1f77b4")

    labels = ["L1: Explicit", "L2: Varied", "L3: Relative", "L4: Distractors", "L5: Noisy"]
    ax.set_xticks(x)
    ax.set_xticklabels(labels[: len(levels)], rotation=15, ha="right")
    ax.set_ylabel("Event F1 (%)")
    ax.set_title("F1 Score Across Difficulty Levels")
    ax.legend()
    ax.grid(axis="y")
    ax.set_ylim(0, 105)

    plt.tight_layout()
    fig.savefig(out_dir / "curriculum_breakdown.pdf")
    fig.savefig(out_dir / "curriculum_breakdown.png", dpi=300)
    plt.close(fig)
    print(f"Saved curriculum breakdown to {out_dir / 'curriculum_breakdown.pdf'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate plots for InboxRL report.")
    parser.add_argument("--log-csv", type=str, default="results/logs/train_metrics.csv")
    parser.add_argument("--out-dir", type=str, default="report/figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    plot_training_curves(Path(args.log_csv), out_dir)


if __name__ == "__main__":
    main()
