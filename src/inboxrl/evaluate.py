"""Evaluation harness: run an extractor over a split and compute metrics with bootstrap CIs.

Usage:
    python -m inboxrl.evaluate --extractor rule --split synth_test
    python -m inboxrl.evaluate --extractor zero-shot --model Qwen/Qwen2.5-0.5B-Instruct --split real_test
    python -m inboxrl.evaluate --extractor agent --model checkpoints/final --split real_test --name grpo_main

Writes results/<name>_<split>.json (metrics) and results/completions/<name>_<split>.jsonl (raw outputs).
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from inboxrl.rewards import RewardWeights, compute_reward_single, title_token_f1, validate_gold
from inboxrl.schema import Event, parse_completion

SPLITS = {
    "synth_val": "data/synthetic/val.jsonl",
    "synth_test": "data/synthetic/test.jsonl",
    "real_test": "data/real/real_test.jsonl",
}
TITLE_TP_THRESHOLD = 0.5


@dataclass
class EmailStats:
    """Per-email counts; micro metrics and bootstrap CIs are computed from these."""
    level: int
    valid: bool
    reward: float
    n_pred: int
    n_gold: int
    tp: int  # matched pairs with correct date and title F1 >= threshold
    paired: int
    date_ok: int
    time_ok: int
    type_ok: int
    title_f1_sum: float
    exact: bool  # every gold event matched with all fields correct and no extra events


@dataclass
class EvaluationMetrics:
    total_samples: int = 0
    valid_json_ratio: float = 0.0
    mean_reward: float = 0.0
    total_pred_events: int = 0
    total_gold_events: int = 0
    true_positive_events: int = 0
    event_precision: float = 0.0
    event_recall: float = 0.0
    event_f1: float = 0.0
    event_f1_ci: list[float] = field(default_factory=list)
    # Fraction of gold events matched by a prediction with the correct date
    date_accuracy: float = 0.0
    date_accuracy_ci: list[float] = field(default_factory=list)
    # Field accuracies over Hungarian-matched pairs
    paired_date_acc: float = 0.0
    paired_time_acc: float = 0.0
    paired_type_acc: float = 0.0
    paired_title_f1: float = 0.0
    exact_match_ratio: float = 0.0
    # No-event emails: correct [] / invalid output / hallucinated events
    no_event_samples: int = 0
    no_event_accuracy: float = 0.0
    no_event_invalid_rate: float = 0.0
    hallucination_rate_on_empty: float = 0.0
    level_f1: dict[int, float] = field(default_factory=dict)
    level_rewards: dict[int, float] = field(default_factory=dict)


def email_stats(record: dict[str, Any], completion: str, weights: RewardWeights) -> EmailStats:
    golds = validate_gold(record["gold"])
    preds = parse_completion(completion)
    reward = compute_reward_single(completion, golds, weights=weights).total
    level = int(record.get("level", 0))
    if preds is None:
        return EmailStats(level, False, reward, 0, len(golds), 0, 0, 0, 0, 0, 0.0, False)

    tp = paired = date_ok = time_ok = type_ok = 0
    title_sum = 0.0
    all_fields_ok = 0
    if preds and golds:
        sim = np.zeros((len(preds), len(golds)))
        for i, p in enumerate(preds):
            for j, g in enumerate(golds):
                sim[i, j] = 0.5 * (p.date == g.date) + 0.5 * title_token_f1(p.title, g.title)
        for r, c in zip(*linear_sum_assignment(-sim)):
            p, g = preds[r], golds[c]
            t_f1 = title_token_f1(p.title, g.title)
            paired += 1
            date_ok += p.date == g.date
            time_ok += p.start_time == g.start_time
            type_ok += p.type == g.type
            title_sum += t_f1
            if p.date == g.date and t_f1 >= TITLE_TP_THRESHOLD:
                tp += 1
                if p.start_time == g.start_time and p.type == g.type:
                    all_fields_ok += 1
    exact = len(preds) == len(golds) and all_fields_ok == len(golds)
    return EmailStats(level, True, reward, len(preds), len(golds), tp, paired, date_ok, time_ok, type_ok, title_sum, exact)


def _f1(tp: int, n_pred: int, n_gold: int) -> tuple[float, float, float]:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def _micro(stats: list[EmailStats]) -> tuple[float, float]:
    tp = sum(s.tp for s in stats)
    f1 = _f1(tp, sum(s.n_pred for s in stats), sum(s.n_gold for s in stats))[2]
    n_gold = sum(s.n_gold for s in stats)
    date_acc = sum(s.date_ok for s in stats) / n_gold if n_gold else 0.0
    return f1, date_acc


def bootstrap_ci(stats: list[EmailStats], n_boot: int = 1000, seed: int = 0) -> tuple[list[float], list[float]]:
    """95% percentile bootstrap CIs for event F1 and date accuracy, resampling emails."""
    rng = np.random.default_rng(seed)
    f1s, dates = [], []
    for _ in range(n_boot):
        sample = [stats[i] for i in rng.integers(0, len(stats), len(stats))]
        f1, d = _micro(sample)
        f1s.append(f1)
        dates.append(d)
    q = lambda xs: [float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5))]  # noqa: E731
    return q(f1s), q(dates)


def evaluate_predictions(
    records: list[dict[str, Any]],
    completions: list[str],
    weights: Optional[RewardWeights] = None,
    n_boot: int = 1000,
) -> EvaluationMetrics:
    """Compute all metrics for completions against gold records."""
    weights = weights or RewardWeights()
    if not records:
        return EvaluationMetrics()
    stats = [email_stats(r, c, weights) for r, c in zip(records, completions, strict=True)]

    n = len(stats)
    tp = sum(s.tp for s in stats)
    n_pred = sum(s.n_pred for s in stats)
    n_gold = sum(s.n_gold for s in stats)
    prec, rec, f1 = _f1(tp, n_pred, n_gold)
    paired = sum(s.paired for s in stats)
    f1_ci, date_ci = bootstrap_ci(stats, n_boot) if n_boot else ([], [])

    empty = [s for s in stats if s.n_gold == 0]
    n_empty = len(empty)

    level_f1, level_rewards = {}, {}
    for lvl in sorted({s.level for s in stats}):
        ls = [s for s in stats if s.level == lvl]
        level_f1[lvl] = _f1(sum(s.tp for s in ls), sum(s.n_pred for s in ls), sum(s.n_gold for s in ls))[2]
        level_rewards[lvl] = float(np.mean([s.reward for s in ls]))

    return EvaluationMetrics(
        total_samples=n,
        valid_json_ratio=sum(s.valid for s in stats) / n,
        mean_reward=float(np.mean([s.reward for s in stats])),
        total_pred_events=n_pred,
        total_gold_events=n_gold,
        true_positive_events=tp,
        event_precision=prec,
        event_recall=rec,
        event_f1=f1,
        event_f1_ci=f1_ci,
        date_accuracy=sum(s.date_ok for s in stats) / n_gold if n_gold else 0.0,
        date_accuracy_ci=date_ci,
        paired_date_acc=sum(s.date_ok for s in stats) / paired if paired else 0.0,
        paired_time_acc=sum(s.time_ok for s in stats) / paired if paired else 0.0,
        paired_type_acc=sum(s.type_ok for s in stats) / paired if paired else 0.0,
        paired_title_f1=sum(s.title_f1_sum for s in stats) / paired if paired else 0.0,
        exact_match_ratio=sum(s.exact for s in stats) / n,
        no_event_samples=n_empty,
        no_event_accuracy=sum(s.valid and s.n_pred == 0 for s in empty) / n_empty if n_empty else 0.0,
        no_event_invalid_rate=sum(not s.valid for s in empty) / n_empty if n_empty else 0.0,
        hallucination_rate_on_empty=sum(s.valid and s.n_pred > 0 for s in empty) / n_empty if n_empty else 0.0,
        level_f1=level_f1,
        level_rewards=level_rewards,
    )


# ---------------------------------------------------------------------------
# Extractors: each maps a list of records to a list of completion strings
# ---------------------------------------------------------------------------

Extractor = Callable[[list[dict[str, Any]]], list[str]]


def empty_extractor(records: list[dict[str, Any]]) -> list[str]:
    """Constant [] policy (reward sanity check)."""
    return ["[]" for _ in records]


def random_extractor(records: list[dict[str, Any]], seed: int = 0) -> list[str]:
    """Random-guess policy: one event with a random type and a random date within 3 weeks."""
    rng = random.Random(seed)
    out = []
    for r in records:
        d = date.fromisoformat(r["received_date"]) + timedelta(days=rng.randint(0, 21))
        ev = {"title": "Event", "type": rng.choice(["deadline", "exam", "class", "event", "meeting"]),
              "date": d.isoformat(), "start_time": None, "end_time": None, "location": None}
        out.append(json.dumps([ev]))
    return out


def rule_extractor(records: list[dict[str, Any]]) -> list[str]:
    from inboxrl.baselines.rule_based import RuleBasedExtractor

    ex = RuleBasedExtractor()
    return [
        json.dumps([e.model_dump() for e in ex.extract_events(r["received_date"], r["sender"], r["subject"], r["body"])])
        for r in records
    ]


def model_extractor(mode: str, model_id: str, batch_size: int, max_new_tokens: int, max_prompt_tokens: int) -> Extractor:
    """zero-shot / few-shot prompting of a base model, or the trained agent (adapter dir)."""
    from inboxrl.agent import generate_batch, load_model
    from inboxrl.baselines.prompting import build_few_shot_messages
    from inboxrl.prompts import build_prompt

    if mode == "agent":
        model, tok = load_model(adapter_path=model_id)
    else:
        model, tok = load_model(base_model_name=model_id)

    def run(records: list[dict[str, Any]]) -> list[str]:
        if mode == "few-shot":
            prompts = [
                tok.apply_chat_template(
                    build_few_shot_messages(r["received_date"], r["sender"], r["subject"], r["body"]),
                    tokenize=False, add_generation_prompt=True,
                )
                for r in records
            ]
        else:  # zero-shot base model and agent use exactly the training prompt
            prompts = [build_prompt(tok, r["received_date"], r["sender"], r["subject"], r["body"], max_prompt_tokens)
                       for r in records]
        return generate_batch(model, tok, prompts, max_new_tokens=max_new_tokens, batch_size=batch_size)

    return run


def load_records(split_or_path: str) -> list[dict[str, Any]]:
    path = Path(SPLITS.get(split_or_path, split_or_path))
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_evaluation(
    records: list[dict[str, Any]],
    extractor: Extractor,
    name: str,
    split: str,
    results_dir: Path = Path("results"),
) -> EvaluationMetrics:
    completions = extractor(records)
    metrics = evaluate_predictions(records, completions)

    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "completions").mkdir(exist_ok=True)
    out_json = results_dir / f"{name}_{split}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"name": name, "split": split, **asdict(metrics)}, f, indent=2)
    with open(results_dir / "completions" / f"{name}_{split}.jsonl", "w", encoding="utf-8") as f:
        for r, c in zip(records, completions):
            f.write(json.dumps({"id": r["id"], "level": r.get("level"), "gold": r["gold"], "completion": c},
                               ensure_ascii=False) + "\n")
    print(f"[{name} | {split}] F1={metrics.event_f1:.3f} {metrics.event_f1_ci} "
          f"date_acc={metrics.date_accuracy:.3f} valid={metrics.valid_json_ratio:.3f} "
          f"reward={metrics.mean_reward:.3f} -> {out_json}")
    return metrics


def build_extractor(args: argparse.Namespace) -> Extractor:
    if args.extractor == "empty":
        return empty_extractor
    if args.extractor == "random":
        return random_extractor
    if args.extractor == "rule":
        return rule_extractor
    if args.model is None:
        raise SystemExit(f"--model is required for --extractor {args.extractor}")
    return model_extractor(args.extractor, args.model, args.batch_size, args.max_new_tokens, args.max_prompt_tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an extractor on an InboxRL split.")
    parser.add_argument("--extractor", required=True, choices=["empty", "random", "rule", "zero-shot", "few-shot", "agent"])
    parser.add_argument("--model", default=None, help="HF model id (zero/few-shot) or adapter dir (agent)")
    parser.add_argument("--split", default="synth_test", help=f"One of {list(SPLITS)} or a JSONL path")
    parser.add_argument("--name", default=None, help="Result name (default: extractor name)")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N emails")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-prompt-tokens", type=int, default=640)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    records = load_records(args.split)[: args.limit]
    split_name = args.split if args.split in SPLITS else Path(args.split).stem
    run_evaluation(records, build_extractor(args), args.name or args.extractor, split_name, Path(args.results_dir))


if __name__ == "__main__":
    main()
