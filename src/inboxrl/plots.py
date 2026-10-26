"""Figures, LaTeX tables and number macros for the report, generated ONLY from real outputs.

Inputs:
    runs/<run>/completions/*.parquet  (logged rollouts; present for every run, including the ones we stopped)
    results/<name>_<split>.json       (written by evaluate.py)
Outputs:
    report/figures/training_story.{pdf,png}   reward, share of `[]` answers and per-level F1
    report/results_table.tex                  one row per baseline / GRPO run
    report/numbers.tex                        \\newcommand macros for every number quoted in the text

Anything missing is skipped with a message; no number is ever filled in by hand.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
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
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 8,
    "font.family": "serif",
    "axes.edgecolor": INK_2,
    "axes.labelcolor": INK,
    "axes.titlesize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "legend.frameon": False,
    "legend.fontsize": 7,
    "lines.linewidth": 1.4,
    "savefig.bbox": "tight",
})

# (result name, label, system-prompt version); baselines first, then our GRPO runs.
BASELINE_ROWS = [
    ("rule", "Regex + dateparser", "--"),
    ("zeroshot_0.5b", "Qwen-0.5B zero-shot", "1"),
    ("zeroshot_0.5b_v2", "Qwen-0.5B zero-shot", "2"),
    ("fewshot_0.5b", "Qwen-0.5B 3-shot", "1"),
    ("zeroshot_1.5b", "Qwen-1.5B zero-shot", "1"),
    ("zeroshot_1.5b_v2", "Qwen-1.5B zero-shot", "2"),
]
GRPO_ROWS = [
    ("grpo_main_s0", "A: baseline reward design, $\\beta{=}.04$", "1"),
    ("grpo_explore_s0", "A + $T{=}1$, $G{=}8$", "1"),
    ("grpo_no_curriculum", "A without curriculum", "1"),
    ("grpo_v4_s0", "B: new prompt, $\\beta{=}.3$", "2"),
    ("grpo_v5_s0", "C: B with $\\beta{=}.15$ (seed 0)", "2"),
    ("grpo_v6_s0", "D: C + 40\\% no-event (seed 0)", "2"),
    ("grpo_v6_s1", "D: C + 40\\% no-event (seed 1)", "2"),
    ("grpo_v6_s2", "D: C + 40\\% no-event (seed 2)", "2"),
]
# groups of seeds of one configuration, summarised as mean / std in numbers.tex
SEED_GROUPS = {"v6": "grpo_v6_s*"}

# runs shown in the training-story figure: (run dir, legend label, colour, linestyle)
STORY_RUNS = [
    ("v2_s0", "v2 ($\\beta{=}.04$)", SERIES[3], "--"),
    ("v3_s0", "v3 (50% no-event, $w_{empty}{=}2$)", INK_2, ":"),
    ("v5_s1", "v5, seed 1", SERIES[1], "--"),
    ("v5_s0", "v5, seed 0", SERIES[0], "-"),
    ("v6_s0", "v6, seed 0", SERIES[2], "-"),
    ("v6_s1", "v6, seed 1", SERIES[2], "-."),
    ("v6_s2", "v6, seed 2", SERIES[2], ":"),
]
LEVEL_NAMES = {1: "L1", 2: "L2", 3: "L3", 4: "L4", 5: "L5"}
LEVEL_SERIES = [
    ("zeroshot_1.5b_v2", "1.5B zero-shot", INK_2),
    ("grpo_v5_s0", "v5 seed 0", SERIES[0]),
    ("grpo_v6_s0", "v6 seed 0", SERIES[2]),
    ("grpo_v6_s1", "v6 seed 1", SERIES[3]),
]


def _save(fig: Any, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf")
    fig.savefig(out_dir / f"{name}.png", dpi=250)
    plt.close(fig)
    print(f"  wrote {out_dir / name}.pdf")


def load_completions(run_dir: Path) -> Optional[pd.DataFrame]:
    files = sorted(glob(str(run_dir / "completions" / "*.parquet")))
    if not files:
        return None
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["is_empty"] = df["completion"].astype(str).str.strip().eq("[]")
    return df


def _binned(df: pd.DataFrame, col: str, width: int) -> pd.Series:
    return df.groupby((df["step"] // width) * width + width // 2)[col].mean()


def load_results(results_dir: Path, split: str) -> dict[str, dict[str, Any]]:
    out = {}
    for path in sorted(results_dir.glob(f"*_{split}.json")):
        name = path.name[: -len(f"_{split}.json")]
        out[name] = json.loads(path.read_text())
    return out


def plot_training_story(runs_dir: Path, results: dict[str, dict[str, Any]], out_dir: Path, width: int = 20) -> None:
    """(a) mean reward, (b) share of `[]` answers, (c) per-level F1; (a, b) come from the logged rollouts."""
    data = {run: load_completions(runs_dir / run) for run, *_ in STORY_RUNS}
    data = {k: v for k, v in data.items() if v is not None}
    if not data:
        print("  no logged rollouts found; skipping training-story figure")
        return
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 1.8), gridspec_kw={"width_ratios": [1, 1, 0.9]})
    ax_r, ax_e, ax_l = axes
    for run, label, colour, ls in STORY_RUNS:
        df = data.get(run)
        if df is None:
            continue
        r, e = _binned(df, "reward_func_total", width), _binned(df, "is_empty", width)
        if run in ("v5_s0", "v6_s0", "v6_s1", "v6_s2"):
            ax_r.plot(r.index, r.values, color=colour, ls=ls, label=label)
        ax_e.plot(e.index, e.values, color=colour, ls=ls, label=label)
    ax_r.set_xlabel("optimizer step")
    ax_r.set_ylabel("mean reward $R$")
    ax_r.set_title("(a) reward of GRPO rollouts")
    ax_r.legend(loc="lower right")
    ax_e.set_xlabel("optimizer step")
    ax_e.set_ylabel("fraction of rollouts")
    ax_e.set_title("(b) share of $[\\,]$ answers")
    ax_e.set_ylim(-0.02, 2.0)
    ax_e.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_e.legend(loc="upper right", fontsize=6, ncol=2, columnspacing=0.8, handlelength=1.6)

    series = [(n, lab, c) for n, lab, c in LEVEL_SERIES if n in results]
    if series:
        x = np.arange(1, 6)
        w = 0.8 / len(series)
        for i, (n, lab, c) in enumerate(series):
            ax_l.bar(x - 0.4 + w * (i + 0.5), [100 * results[n]["level_f1"].get(str(l), 0.0) for l in x], w,
                     color=c, label=lab)
        ax_l.set_xticks(x)
        ax_l.set_xticklabels([LEVEL_NAMES[int(l)] for l in x])
        ax_l.set_ylabel("event F1 (%)")
        ax_l.set_ylim(0, 135)
        ax_l.set_yticks(range(0, 101, 20))
        ax_l.grid(axis="x", visible=False)
        ax_l.set_title("(c) F1 by difficulty level")
        ax_l.legend(loc="upper right", ncol=2, fontsize=6, columnspacing=0.8, handlelength=1.0)
    fig.tight_layout(w_pad=1.0)
    _save(fig, out_dir, "training_story")


def _pct(x: float) -> str:
    return f"{100 * x:.1f}"


def _row(label: str, prompt: str, m: dict[str, Any]) -> str:
    ci = m.get("event_f1_ci") or [m["event_f1"]] * 2
    l3 = m.get("level_f1", {}).get("3")
    return (f"{label} & {prompt} & {_pct(m['event_f1'])} \\scriptsize[{_pct(ci[0])}, {_pct(ci[1])}] & "
            f"{_pct(m['date_accuracy'])} & {_pct(m['valid_json_ratio'])} & "
            f"{_pct(m['no_event_accuracy']) if 'no_event_accuracy' in m else '--'} & "
            f"{_pct(l3) if l3 is not None else '--'} \\\\")


def write_results_table(results: dict[str, dict[str, Any]], out: Path) -> None:
    if not results:
        print("  no results; skipping results table")
        return
    lines = [
        "% AUTO-GENERATED by `python -m inboxrl.plots` from results/*.json -- do not edit by hand.",
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "Method & Prompt & F1 \\scriptsize[95\\% CI] & Date & JSON & No-event & L3 F1 \\\\",
        "\\midrule",
    ]
    for block in (BASELINE_ROWS, GRPO_ROWS):
        rows = [_row(label, prompt, results[name]) for name, label, prompt in block if name in results]
        lines += rows
        if block is BASELINE_ROWS and rows:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    out.write_text("\n".join(lines) + "\n")
    print(f"  wrote {out}")


_DIGITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def _macro(*parts: str) -> str:
    """LaTeX control sequences may only contain letters: spell digits out, drop everything else."""
    text = "_".join(parts)
    text = re.sub(r"\d", lambda m: "_" + _DIGITS[int(m.group())] + "_", text)
    return "".join(p.capitalize() for p in re.split(r"[^A-Za-z]+", text) if p)


def l3_date_errors(comp_dir: Path, name: str, split: str) -> Optional[dict[str, float]]:
    """Signed day error of the predicted date on L3 (relative-date) emails whose output parsed as a JSON list."""
    path = comp_dir / f"{name}_{split}.jsonl"
    if not path.exists():
        return None
    errs: list[int] = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["level"] != 3:
            continue
        try:
            pred = json.loads(row["completion"])
            errs += [(dt.date.fromisoformat(p["date"]) - dt.date.fromisoformat(g["date"])).days
                     for g, p in zip(row["gold"], pred)]
        except (ValueError, KeyError, TypeError):
            continue
    if not errs:
        return None
    a = np.abs(np.array(errs))
    return {"N": float(len(a)), "Exact": 100 * float((a == 0).mean()), "OffOne": 100 * float((a == 1).mean()),
            "OffFar": 100 * float((a >= 4).mean())}


def write_numbers(results: dict[str, dict[str, Any]], out: Path, comp_dir: Optional[Path] = None, split: str = "synth_test") -> None:
    """One macro per quoted number, e.g. \\nmGrpoVfiveSzeroF = 72.2. The report text only uses these."""
    lines = ["% AUTO-GENERATED by `python -m inboxrl.plots` from results/*.json -- do not edit by hand."]

    def emit(name: str, value: float) -> None:
        lines.append(f"\\newcommand{{\\nm{name}}}{{{value:.1f}}}")

    fields = {"F": "event_f1", "Date": "date_accuracy", "Json": "valid_json_ratio", "NoEv": "no_event_accuracy",
              "Prec": "event_precision", "Rec": "event_recall", "TitleF": "paired_title_f1",
              "TimeAcc": "paired_time_acc"}
    for name, m in results.items():
        base = _macro(name)
        for short, key in fields.items():
            if key in m:
                emit(base + short, 100 * m[key])
        ci = m.get("event_f1_ci")
        if ci:
            emit(base + "FLo", 100 * ci[0])
            emit(base + "FHi", 100 * ci[1])
        for lvl, v in m.get("level_f1", {}).items():
            emit(base + "Lvl" + _DIGITS[int(lvl)].capitalize(), 100 * v)
    for name in results:
        stats = l3_date_errors(comp_dir, name, split) if comp_dir is not None and name.startswith("grpo_") else None
        for short, v in (stats or {}).items():
            emit(_macro(name) + "LthreeDate" + short, v)
    for group, pattern in SEED_GROUPS.items():
        members = [m for n, m in sorted(results.items()) if re.fullmatch(pattern.replace("*", ".*"), n)]
        if len(members) >= 2:
            lines.append(f"\\newcommand{{\\nm{_macro(group)}Seeds}}{{{len(members)}}}")
            for short, key in {"F": "event_f1", "Date": "date_accuracy", "Json": "valid_json_ratio"}.items():
                arr = np.array([m[key] for m in members]) * 100
                emit(_macro(group) + "Mean" + short, float(arr.mean()))
                emit(_macro(group) + "Std" + short, float(arr.std()))
    out.write_text("\n".join(lines) + "\n")
    print(f"  wrote {out} ({len(lines) - 1} macros)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate report figures, tables and macros from real runs/results.")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--split", default="synth_test")
    parser.add_argument("--out-dir", default="report/figures")
    parser.add_argument("--tables-dir", default="report")
    args = parser.parse_args()

    results = load_results(Path(args.results_dir), args.split)
    tables_dir = Path(args.tables_dir)
    plot_training_story(Path(args.runs_dir), results, Path(args.out_dir))
    write_results_table(results, tables_dir / "results_table.tex")
    write_numbers(results, tables_dir / "numbers.tex", Path(args.results_dir) / "completions", args.split)


if __name__ == "__main__":
    main()
