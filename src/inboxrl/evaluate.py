"""Evaluation harness and metrics computation for InboxRL."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment

from inboxrl.schema import Event, parse_completion
from inboxrl.rewards import compute_reward_single, RewardWeights


@dataclass
class EvaluationMetrics:
    total_samples: int = 0
    valid_json_count: int = 0
    valid_json_ratio: float = 0.0
    mean_reward: float = 0.0
    # Event-level detection metrics (Hungarian paired with date exact match + title >= 0.5)
    total_pred_events: int = 0
    total_gold_events: int = 0
    true_positive_events: int = 0
    event_precision: float = 0.0
    event_recall: float = 0.0
    event_f1: float = 0.0
    # Field-level accuracies on matched pairs
    date_match_ratio: float = 0.0
    time_match_ratio: float = 0.0
    type_match_ratio: float = 0.0
    mean_title_similarity: float = 0.0
    # No-event email metrics
    no_event_samples: int = 0
    no_event_correct: int = 0
    no_event_accuracy: float = 0.0
    hallucination_rate_on_empty: float = 0.0
    # Breakdown by difficulty level
    level_f1: dict[int, float] = field(default_factory=dict)
    level_rewards: dict[int, float] = field(default_factory=dict)


def evaluate_predictions(
    records: list[dict[str, Any]],
    completions: list[str],
    weights: Optional[RewardWeights] = None,
) -> EvaluationMetrics:
    """Evaluate completions against gold records."""
    if weights is None:
        weights = RewardWeights()

    n = len(records)
    if n == 0:
        return EvaluationMetrics()

    valid_json_count = 0
    rewards = []
    total_preds = 0
    total_golds = 0
    total_tp = 0

    matched_date_correct = 0
    matched_time_correct = 0
    matched_type_correct = 0
    title_sims = []
    total_paired = 0

    no_event_samples = 0
    no_event_correct = 0

    level_golds: dict[int, int] = {}
    level_preds: dict[int, int] = {}
    level_tps: dict[int, int] = {}
    level_rewards_accum: dict[int, list[float]] = {}

    for record, comp in zip(records, completions):
        level = record.get("level", 1)
        gold_raw = record.get("gold", [])
        golds = [Event.model_validate(g) for g in gold_raw if isinstance(g, dict)]

        total_golds += len(golds)
        level_golds[level] = level_golds.get(level, 0) + len(golds)

        if len(golds) == 0:
            no_event_samples += 1

        preds = parse_completion(comp)
        breakdown = compute_reward_single(comp, golds, weights=weights)
        rewards.append(breakdown.total)
        level_rewards_accum.setdefault(level, []).append(breakdown.total)

        if preds is not None:
            valid_json_count += 1
            total_preds += len(preds)
            level_preds[level] = level_preds.get(level, 0) + len(preds)

            if len(golds) == 0 and len(preds) == 0:
                no_event_correct += 1

            if len(preds) > 0 and len(golds) > 0:
                # Run Hungarian matching
                cost_matrix = np.zeros((len(preds), len(golds)), dtype=np.float32)
                for i, p in enumerate(preds):
                    for j, g in enumerate(golds):
                        # Similarity for matching
                        d_ok = 1.0 if p.date == g.date else 0.0
                        t_sim = fuzz.token_set_ratio(p.title.lower(), g.title.lower()) / 100.0
                        sim = 0.5 * d_ok + 0.5 * t_sim
                        cost_matrix[i, j] = 1.0 - sim

                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                for r, c in zip(row_ind, col_ind):
                    p, g = preds[r], golds[c]
                    t_sim = fuzz.token_set_ratio(p.title.lower(), g.title.lower()) / 100.0
                    date_ok = (p.date == g.date)

                    # A true positive requires correct date and plausible title
                    if date_ok and t_sim >= 0.5:
                        total_tp += 1
                        level_tps[level] = level_tps.get(level, 0) + 1

                    total_paired += 1
                    if date_ok:
                        matched_date_correct += 1
                    if p.start_time == g.start_time:
                        matched_time_correct += 1
                    if p.type == g.type:
                        matched_type_correct += 1
                    title_sims.append(t_sim)
        else:
            if len(golds) == 0:
                pass  # invalid json is counted as failure

    prec = total_tp / total_preds if total_preds > 0 else 0.0
    rec = total_tp / total_golds if total_golds > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    level_f1 = {}
    for lvl in level_rewards_accum.keys():
        lvl_g = level_golds.get(lvl, 0)
        lvl_p = level_preds.get(lvl, 0)
        lvl_tp = level_tps.get(lvl, 0)
        l_prec = lvl_tp / lvl_p if lvl_p > 0 else 0.0
        l_rec = lvl_tp / lvl_g if lvl_g > 0 else 0.0
        level_f1[lvl] = (2 * l_prec * l_rec / (l_prec + l_rec)) if (l_prec + l_rec) > 0 else 0.0

    level_rewards = {
        lvl: float(np.mean(r_list)) for lvl, r_list in level_rewards_accum.items()
    }

    no_event_acc = no_event_correct / no_event_samples if no_event_samples > 0 else 1.0
    hallucination_rate = 1.0 - no_event_acc

    return EvaluationMetrics(
        total_samples=n,
        valid_json_count=valid_json_count,
        valid_json_ratio=valid_json_count / n,
        mean_reward=float(np.mean(rewards)),
        total_pred_events=total_preds,
        total_gold_events=total_golds,
        true_positive_events=total_tp,
        event_precision=prec,
        event_recall=rec,
        event_f1=f1,
        date_match_ratio=matched_date_correct / total_paired if total_paired > 0 else 0.0,
        time_match_ratio=matched_time_correct / total_paired if total_paired > 0 else 0.0,
        type_match_ratio=matched_type_correct / total_paired if total_paired > 0 else 0.0,
        mean_title_similarity=float(np.mean(title_sims)) if title_sims else 0.0,
        no_event_samples=no_event_samples,
        no_event_correct=no_event_correct,
        no_event_accuracy=no_event_acc,
        hallucination_rate_on_empty=hallucination_rate,
        level_f1=level_f1,
        level_rewards=level_rewards,
    )


def run_evaluation(
    data_path: Path,
    extractor_func: Callable[[dict[str, Any]], str],
    out_json: Optional[Path] = None,
) -> EvaluationMetrics:
    with open(data_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    completions = []
    for r in records:
        comp = extractor_func(r)
        completions.append(comp)

    metrics = evaluate_predictions(records, completions)

    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(asdict(metrics), f, indent=2)
        print(f"Results saved to {out_json}")

    return metrics
