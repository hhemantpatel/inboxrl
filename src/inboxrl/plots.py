"""Figures and LaTeX tables for the report, generated ONLY from real outputs.

Inputs:
    runs/<run>/metrics.csv          (written by train_grpo.py)
    results/<name>_<split>.json     (written by evaluate.py)
Outputs:
    report/figures/*.pdf|png, report/results_table.tex, report/ablation_table.tex

Anything missing is skipped with a message; no number is ever filled in by hand.

Result-name convention (see README):
    baselines: empty, random, rule, zeroshot_0.5b, fewshot_0.5b, zeroshot_1.5b
    GRPO main: grpo_main_s0, grpo_main_s1, grpo_main_s2 (seeds)
    ablations: grpo_no_curriculum, grpo_beta0, grpo_format_only, grpo_G2
"""

from __future__ import annotations

import argparse
import json
from glob import glob
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Validated categorical palette (fixed order) + neutral ink for text, grid, baselines
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK_2, GRID, NEUTRAL = "#0b0b0b", "#52514e", "#e4e3df", "#9b9a95"

plt.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "axes.edgecolor": INK_2,
    "axes.labelcolor": INK,
    "axes.titlesize": 9.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "lines.linewidth": 1.6,
    "savefig.bbox": "tight",
})

BASELINE_LABELS = {
    "empty": "Constant []",
    "random": "Random guess",
    "rule": "Regex + dateparser",
    "zeroshot_0.5b": "Qwen-0.5B zero-shot",
    "fewshot_0.5b": "Qwen-0.5B 3-shot",
    "zeroshot_1.5b": "Qwen-1.5B zero-shot",
}
ABLATION_LABELS = {
    "grpo_no_curriculum": "w/o curriculum",
    "grpo_beta0": r"w/o KL ($\beta=0$)",
    "grpo_format_only": "format reward only",
    "grpo_G2": "group size $G=2$",
}
MAIN_PREFIX = "grpo_main_s"


def _save(fig: Any, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf")
    fig.savefig(out_dir / f"{name}.png", dpi=250)
    plt.close(fig)
    print(f"  wrote {out_dir / name}.pdf")


def load_runs(pattern: str) -> dict[str, pd.DataFrame]:
    runs = {}
    for path in sorted(glob(str(Path(pattern) / "metrics.csv"))):
        df = pd.read_csv(path)
        runs[Path(path).parent.name] = df
    return runs


def _train_rows(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return df[["step", col]].dropna() if col in df else pd.DataFrame(columns=["step", col])


def _mean_std(runs: dict[str, pd.DataFrame], col: str, smooth: int) -> Optional[pd.DataFrame]:
    """Align runs on step, smooth each with a rolling mean, return mean/std across runs."""
    series = []
    for df in runs.values():
        rows = _train_rows(df, col)
        if rows.empty:
            continue
        s = rows.groupby("step")[col].mean().rolling(smooth, min_periods=1).mean()
        series.append(s)
    if not series:
        return None
    joined = pd.concat(series, axis=1).dropna()
    return pd.DataFrame({"mean": joined.mean(axis=1), "std": joined.std(axis=1).fillna(0.0)})


def plot_training(runs: dict[str, pd.DataFrame], out_dir: Path, smooth: int = 5) -> None:
    if not runs:
        print("  no training runs found; skipping training curves")
        return
    n = len(runs)
    seeds_note = f"mean $\\pm$ std over {n} seeds" if n > 1 else "single run"

    # 1. Total reward (train, smoothed) + validation reward
    stats = _mean_std(runs, "reward", smooth)
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    if stats is not None:
        ax.plot(stats.index, stats["mean"], color=SERIES[0], label="train (rollouts)")
        ax.fill_between(stats.index, stats["mean"] - stats["std"], stats["mean"] + stats["std"],
                        color=SERIES[0], alpha=0.18, linewidth=0)
    ev = _mean_std(runs, "eval_reward", 1)
    if ev is not None:
        ax.errorbar(ev.index, ev["mean"], yerr=ev["std"] if n > 1 else None, color=SERIES[1],
                    marker="o", markersize=4, linewidth=1.2, capsize=2, label="validation")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("reward $R$")
    ax.set_title(f"GRPO reward ({seeds_note})")
    ax.legend(loc="lower right")
    _save(fig, out_dir, "reward_vs_step")

    # 2. Reward components (same scale family: each in [-1, 1])
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for i, (col, label) in enumerate([("rewards/reward_func_format/mean", "$r_{fmt}$"),
                                      ("rewards/reward_func_match/mean", "$r_{match}$"),
                                      ("rewards/reward_func_empty/mean", "$r_{empty}$"),
                                      ("rewards/reward_func_len/mean", "$r_{len}$")]):
        s = _mean_std(runs, col, smooth)
        if s is not None:
            ax.plot(s.index, s["mean"], color=SERIES[i], label=label)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("component value")
    ax.set_title("Reward decomposition")
    ax.legend(loc="lower right", ncol=2)
    _save(fig, out_dir, "reward_components")

    # 3. Diagnostics as small multiples (separate y-axes, never a dual axis)
    diag = [("kl", "KL to reference"), ("completions/mean_length", "completion tokens"),
            ("frac_reward_zero_std", "groups with zero reward std")]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.9))
    for ax, (col, title) in zip(axes, diag):
        s = _mean_std(runs, col, smooth)
        if s is not None:
            ax.plot(s.index, s["mean"], color=SERIES[0])
            ax.fill_between(s.index, s["mean"] - s["std"], s["mean"] + s["std"], color=SERIES[0], alpha=0.18, linewidth=0)
        ax.set_title(title)
        ax.set_xlabel("step")
    _save(fig, out_dir, "training_diagnostics")


def load_results(results_dir: Path, split: str) -> dict[str, dict[str, Any]]:
    out = {}
    for path in sorted(results_dir.glob(f"*_{split}.json")):
        name = path.name[: -len(f"_{split}.json")]
        out[name] = json.loads(path.read_text())
    return out


def _grpo_main(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [v for k, v in sorted(results.items()) if k.startswith(MAIN_PREFIX)]


def plot_comparison(results: dict[str, dict[str, Any]], split: str, out_dir: Path) -> None:
    names = [n for n in BASELINE_LABELS if n in results and n not in ("empty", "random")]
    mains = _grpo_main(results)
    if not names and not mains:
        print(f"  no results for {split}; skipping comparison plot")
        return
    labels = [BASELINE_LABELS[n] for n in names]
    rows = [results[n] for n in names]
    if mains:
        labels.append("InboxRL (GRPO)")
        rows.append(mains[0] if len(mains) == 1 else None)  # placeholder, filled below

    metrics = [("event_f1", "Event F1"), ("date_accuracy", "Date accuracy"), ("valid_json_ratio", "Valid JSON")]
    x = np.arange(len(labels))
    width = 0.26
    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    for i, (key, mlabel) in enumerate(metrics):
        vals, errs = [], [[], []]
        for j, row in enumerate(rows):
            if row is None:  # GRPO over several seeds: mean and std
                arr = np.array([m[key] for m in mains])
                v, lo, hi = arr.mean(), arr.std(), arr.std()
            else:
                v = row[key]
                ci = row.get(f"{key}_ci") or []
                lo, hi = (v - ci[0], ci[1] - v) if len(ci) == 2 else (0.0, 0.0)
            vals.append(100 * v)
            errs[0].append(100 * lo)
            errs[1].append(100 * hi)
        ax.bar(x + (i - 1) * width, vals, width - 0.03, color=SERIES[i], label=mlabel,
               yerr=errs, error_kw={"elinewidth": 0.8, "capsize": 2, "ecolor": INK_2})
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("%")
    ax.set_ylim(0, 122)  # headroom so the legend never covers a bar
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis="x", visible=False)
    ax.set_title(f"Comparison on {split.replace('_', ' ')} (error bars: 95% bootstrap CI; GRPO: std over seeds)")
    ax.legend(loc="upper left", ncol=3)
    _save(fig, out_dir, f"comparison_{split}")


def plot_levels(results: dict[str, dict[str, Any]], split: str, out_dir: Path) -> None:
    base = results.get("zeroshot_0.5b")
    mains = _grpo_main(results)
    if base is None or not mains:
        print(f"  need zeroshot_0.5b and grpo_main_s* results on {split}; skipping per-level plot")
        return
    levels = sorted(int(k) for k in base["level_f1"])
    x = np.arange(len(levels))
    grpo = np.array([[m["level_f1"].get(str(l), 0.0) for l in levels] for m in mains])
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    ax.bar(x - 0.2, [100 * base["level_f1"][str(l)] for l in levels], 0.37, color=NEUTRAL, label="zero-shot base")
    ax.bar(x + 0.2, 100 * grpo.mean(0), 0.37, color=SERIES[0], label="InboxRL (GRPO)",
           yerr=100 * grpo.std(0) if len(mains) > 1 else None, error_kw={"elinewidth": 0.8, "capsize": 2})
    names = {1: "L1 explicit", 2: "L2 formats", 3: "L3 relative", 4: "L4 distract.", 5: "L5 noisy"}
    ax.set_xticks(x)
    ax.set_xticklabels([names.get(l, f"L{l}") for l in levels], rotation=20, ha="right")
    ax.set_ylabel("event F1 (%)")
    ax.set_ylim(0, 125)
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis="x", visible=False)
    ax.set_title("F1 by difficulty level")
    ax.legend(loc="upper left", ncol=2)
    _save(fig, out_dir, f"levels_{split}")


def _fmt(m: dict[str, Any], key: str) -> str:
    return f"{100 * m[key]:.1f}"


def _fmt_seeds(mains: list[dict[str, Any]], key: str) -> str:
    arr = np.array([m[key] for m in mains]) * 100
    return f"{arr.mean():.1f}" if len(arr) == 1 else f"{arr.mean():.1f}$\\pm${arr.std():.1f}"


def write_results_table(results_by_split: dict[str, dict[str, dict[str, Any]]], out: Path) -> None:
    """One row per method; F1 / date accuracy / valid JSON / mean reward for each split present."""
    splits = [s for s, r in results_by_split.items() if r]
    if not splits:
        print("  no results; skipping results table")
        return
    cols = " & ".join(f"\\multicolumn{{3}}{{c}}{{{s.replace('_', ' ')}}}" for s in splits)
    sub = " & ".join(["F1 & Date & JSON"] * len(splits))
    lines = [
        "% AUTO-GENERATED by `python -m inboxrl.plots` from results/*.json -- do not edit by hand.",
        f"\\begin{{tabular}}{{l{'ccc' * len(splits)}}}",
        "\\toprule",
        f" & {cols} \\\\",
        f"Method & {sub} \\\\",
        "\\midrule",
    ]
    for name, label in BASELINE_LABELS.items():
        cells = []
        for s in splits:
            m = results_by_split[s].get(name)
            cells += [_fmt(m, "event_f1"), _fmt(m, "date_accuracy"), _fmt(m, "valid_json_ratio")] if m else ["--"] * 3
        if any(c != "--" for c in cells):
            lines.append(f"{label} & {' & '.join(cells)} \\\\")
    cells = []
    for s in splits:
        mains = _grpo_main(results_by_split[s])
        cells += ([_fmt_seeds(mains, k) for k in ("event_f1", "date_accuracy", "valid_json_ratio")]
                  if mains else ["--"] * 3)
    lines += ["\\midrule", f"InboxRL (GRPO) & {' & '.join(cells)} \\\\", "\\bottomrule", "\\end{tabular}"]
    out.write_text("\n".join(lines) + "\n")
    print(f"  wrote {out}")


def write_ablation_table(results: dict[str, dict[str, Any]], split: str, out: Path) -> None:
    mains = _grpo_main(results)
    rows = [(label, results[name]) for name, label in ABLATION_LABELS.items() if name in results]
    if not mains or not rows:
        print(f"  need grpo_main_s* and ablation results on {split}; skipping ablation table")
        return
    lines = [
        "% AUTO-GENERATED by `python -m inboxrl.plots` from results/*.json -- do not edit by hand.",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Configuration & F1 & Date & JSON & $\\bar R$ \\\\",
        "\\midrule",
        "Full InboxRL & " + " & ".join(_fmt_seeds(mains, k) for k in ("event_f1", "date_accuracy", "valid_json_ratio"))
        + f" & {np.mean([m['mean_reward'] for m in mains]):.2f} \\\\",
    ]
    for label, m in rows:
        lines.append(f"{label} & {_fmt(m, 'event_f1')} & {_fmt(m, 'date_accuracy')} & "
                     f"{_fmt(m, 'valid_json_ratio')} & {m['mean_reward']:.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    out.write_text("\n".join(lines) + "\n")
    print(f"  wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate report figures and tables from real runs/results.")
    parser.add_argument("--main-runs", default="runs/main_s*", help="Glob of main-config run dirs (one per seed)")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--splits", nargs="+", default=["synth_test", "real_test"])
    parser.add_argument("--ablation-split", default="synth_test")
    parser.add_argument("--out-dir", default="report/figures")
    parser.add_argument("--tables-dir", default="report")
    args = parser.parse_args()

    out_dir, tables_dir, results_dir = Path(args.out_dir), Path(args.tables_dir), Path(args.results_dir)
    plot_training(load_runs(args.main_runs), out_dir)

    results_by_split = {s: load_results(results_dir, s) for s in args.splits}
    for split, results in results_by_split.items():
        plot_comparison(results, split, out_dir)
        plot_levels(results, split, out_dir)
    write_results_table(results_by_split, tables_dir / "results_table.tex")
    write_ablation_table(results_by_split.get(args.ablation_split, {}), args.ablation_split,
                         tables_dir / "ablation_table.tex")


if __name__ == "__main__":
    main()
