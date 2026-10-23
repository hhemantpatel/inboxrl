"""Reward functions and Hungarian bipartite matching for InboxRL."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

import numpy as np
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment

from inboxrl.schema import Event, parse_completion


@dataclass
class RewardWeights:
    w_format: float = 0.5
    w_match: float = 2.0
    w_empty: float = 1.0
    w_len: float = 0.1
    # Pair-wise sub-weights (sum to 1.0)
    w_date: float = 0.35
    w_time: float = 0.20
    w_title: float = 0.25
    w_type: float = 0.10
    w_loc: float = 0.10


@dataclass
class RewardBreakdown:
    total: float
    r_format: float
    r_match: float
    r_empty: float
    r_len: float
    num_pred: int
    num_gold: int
    num_matched: int


def _event_similarity(pred: Event, gold: Event, weights: RewardWeights) -> float:
    """Compute similarity S(pred, gold) in [0.0, 1.0]."""
    # 1. Date exact match
    date_sim = 1.0 if pred.date == gold.date else 0.0

    # 2. Time match (both None counts as 1.0)
    if pred.start_time is None and gold.start_time is None:
        time_sim = 1.0
    elif pred.start_time is not None and gold.start_time is not None:
        time_sim = 1.0 if pred.start_time == gold.start_time else 0.0
    else:
        time_sim = 0.0

    # 3. Title fuzzy similarity (token_set_ratio is robust to phrasing & reordering)
    title_score = fuzz.token_set_ratio(pred.title.lower(), gold.title.lower()) / 100.0
    # Also boost if exact substring
    if pred.title.lower() in gold.title.lower() or gold.title.lower() in pred.title.lower():
        title_score = max(title_score, 0.9)

    # 4. Type exact match
    type_sim = 1.0 if pred.type == gold.type else 0.0

    # 5. Location similarity
    if pred.location is None and gold.location is None:
        loc_sim = 1.0
    elif pred.location is not None and gold.location is not None:
        loc_sim = fuzz.ratio(pred.location.lower(), gold.location.lower()) / 100.0
    else:
        loc_sim = 0.0

    return (
        weights.w_date * date_sim
        + weights.w_time * time_sim
        + weights.w_title * title_score
        + weights.w_type * type_sim
        + weights.w_loc * loc_sim
    )


def compute_hungarian_match_score(
    preds: list[Event],
    golds: list[Event],
    weights: RewardWeights,
) -> tuple[float, int]:
    """Compute normalized bipartite match score using Hungarian algorithm.
    
    Returns:
        (r_match, num_matched)
    """
    n_pred = len(preds)
    n_gold = len(golds)

    if n_pred == 0 or n_gold == 0:
        return 0.0, 0

    # Construct similarity matrix (n_pred x n_gold)
    sim_matrix = np.zeros((n_pred, n_gold), dtype=np.float32)
    for i, p in enumerate(preds):
        for j, g in enumerate(golds):
            sim_matrix[i, j] = _event_similarity(p, g, weights)

    # Cost matrix for linear sum assignment (minimization problem)
    cost_matrix = 1.0 - sim_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_sim = float(sim_matrix[row_ind, col_ind].sum())
    # Count pairs with significant match (similarity >= 0.5)
    num_matched = int((sim_matrix[row_ind, col_ind] >= 0.5).sum())

    # Normalization: divide by max(|pred|, |gold|)
    # This penalizes both hallucinated extra events and missed events
    norm_factor = max(n_pred, n_gold)
    r_match = matched_sim / norm_factor

    return r_match, num_matched


def _ensure_event_list(raw_gold: Any) -> list[Event]:
    """Coerce various gold formats (JSON string, dict list, Event list) into list[Event]."""
    if isinstance(raw_gold, list):
        out = []
        for item in raw_gold:
            if isinstance(item, Event):
                out.append(item)
            elif isinstance(item, dict):
                try:
                    out.append(Event.model_validate(item))
                except Exception:
                    pass
        return out
    elif isinstance(raw_gold, str):
        parsed = parse_completion(raw_gold)
        return parsed if parsed is not None else []
    return []


def compute_reward_single(
    completion: str,
    gold: Union[list[dict], list[Event], str],
    prompt: Optional[str] = None,
    weights: Optional[RewardWeights] = None,
) -> RewardBreakdown:
    """Compute detailed reward breakdown for a single model completion."""
    if weights is None:
        weights = RewardWeights()

    golds = _ensure_event_list(gold)
    preds = parse_completion(completion)

    # Format reward
    if preds is None:
        # Invalid JSON or failed schema validation
        r_format = -1.0
        r_match = 0.0
        r_empty = -1.0 if len(golds) == 0 else -0.5
        r_len = -0.5
        total = weights.w_format * r_format + weights.w_empty * r_empty + weights.w_len * r_len
        return RewardBreakdown(
            total=total,
            r_format=r_format,
            r_match=r_match,
            r_empty=r_empty,
            r_len=r_len,
            num_pred=0,
            num_gold=len(golds),
            num_matched=0,
        )

    r_format = 1.0
    n_pred = len(preds)
    n_gold = len(golds)

    # Empty vs Non-empty reward logic
    if n_gold == 0:
        if n_pred == 0:
            # Gold is empty and model correctly produced []
            r_empty = 1.0
            r_match = 0.0
            num_matched = 0
        else:
            # Model hallucinated events when email had none
            r_empty = -1.0
            r_match = 0.0
            num_matched = 0
    else:
        if n_pred == 0:
            # Model returned [] when events actually existed
            r_empty = -0.5
            r_match = 0.0
            num_matched = 0
        else:
            r_empty = 0.0
            r_match, num_matched = compute_hungarian_match_score(preds, golds, weights)

    # Length / cleanliness penalty
    raw = completion.strip()
    # Check if there is trailing prose outside JSON brackets
    has_prose = False
    start_bracket = raw.find("[")
    end_bracket = raw.rfind("]")
    if start_bracket > 15 or (end_bracket != -1 and len(raw) - end_bracket > 15):
        has_prose = True

    r_len = -0.3 if has_prose else 0.0
    # Small penalty if unusually long (> 800 chars) for JSON
    if len(raw) > 800:
        r_len -= 0.2

    total = (
        weights.w_format * r_format
        + weights.w_match * r_match
        + weights.w_empty * r_empty
        + weights.w_len * r_len
    )

    return RewardBreakdown(
        total=total,
        r_format=r_format,
        r_match=r_match,
        r_empty=r_empty,
        r_len=r_len,
        num_pred=n_pred,
        num_gold=n_gold,
        num_matched=num_matched,
    )


# ==============================================================================
# TRL GRPOTrainer compatible reward functions
# ==============================================================================

def reward_func_total(prompts: Sequence[Any], completions: Sequence[Any], **kwargs: Any) -> list[float]:
    """Overall weighted reward function for GRPOTrainer."""
    gold_list = kwargs.get("gold", [])
    rewards: list[float] = []
    for i, comp in enumerate(completions):
        comp_str = comp if isinstance(comp, str) else str(comp)
        gold = gold_list[i] if i < len(gold_list) else []
        prompt_str = prompts[i] if i < len(prompts) else None
        res = compute_reward_single(comp_str, gold, prompt=prompt_str)
        rewards.append(res.total)
    return rewards


def reward_func_format(prompts: Sequence[Any], completions: Sequence[Any], **kwargs: Any) -> list[float]:
    """Format validity reward (+1 for valid JSON list of Events, -1 otherwise)."""
    rewards: list[float] = []
    for comp in completions:
        comp_str = comp if isinstance(comp, str) else str(comp)
        preds = parse_completion(comp_str)
        rewards.append(1.0 if preds is not None else -1.0)
    return rewards


def reward_func_match(prompts: Sequence[Any], completions: Sequence[Any], **kwargs: Any) -> list[float]:
    """Hungarian matching reward component."""
    gold_list = kwargs.get("gold", [])
    rewards: list[float] = []
    for i, comp in enumerate(completions):
        comp_str = comp if isinstance(comp, str) else str(comp)
        gold = gold_list[i] if i < len(gold_list) else []
        res = compute_reward_single(comp_str, gold)
        rewards.append(res.r_match)
    return rewards


def reward_func_empty(prompts: Sequence[Any], completions: Sequence[Any], **kwargs: Any) -> list[float]:
    """Empty precision/recall reward component."""
    gold_list = kwargs.get("gold", [])
    rewards: list[float] = []
    for i, comp in enumerate(completions):
        comp_str = comp if isinstance(comp, str) else str(comp)
        gold = gold_list[i] if i < len(gold_list) else []
        res = compute_reward_single(comp_str, gold)
        rewards.append(res.r_empty)
    return rewards
