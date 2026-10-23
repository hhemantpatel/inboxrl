"""Reward functions and Hungarian bipartite matching for InboxRL."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

import numpy as np
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment

from inboxrl.schema import Event, extract_json_payload, parse_completion


@dataclass
class RewardWeights:
    w_format: float = 0.5
    w_match: float = 2.0
    w_empty: float = 1.0
    w_len: float = 0.25
    # Pair-wise sub-weights (sum to 1.0)
    w_date: float = 0.35
    w_time: float = 0.20
    w_title: float = 0.25
    w_type: float = 0.10
    w_loc: float = 0.10
    # Fraction of the non-date fields' credit kept when the date is wrong.
    # A calendar entry on the wrong day is mostly useless, so a wrong date
    # caps the pair similarity at 0.25 * 0.65 ~= 0.16.
    wrong_date_factor: float = 0.25
    max_chars: int = 800


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


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def title_token_f1(pred: str, gold: str) -> float:
    """Bag-of-tokens F1 between two titles.

    Unlike rapidfuzz.token_set_ratio, this cannot be gamed by emitting a
    one-word title that happens to be a subset of the gold title: recall drops.
    """
    p, g = _tokens(pred), _tokens(gold)
    if not p or not g:
        return 0.0
    g_counts: dict[str, int] = {}
    for t in g:
        g_counts[t] = g_counts.get(t, 0) + 1
    common = 0
    for t in p:
        if g_counts.get(t, 0) > 0:
            common += 1
            g_counts[t] -= 1
    if common == 0:
        return 0.0
    precision = common / len(p)
    recall = common / len(g)
    return 2 * precision * recall / (precision + recall)


def _event_similarity(pred: Event, gold: Event, weights: RewardWeights) -> float:
    """Compute similarity S(pred, gold) in [0.0, 1.0]."""
    date_ok = pred.date == gold.date

    # Time match (both None counts as a match)
    time_sim = 1.0 if pred.start_time == gold.start_time else 0.0

    title_sim = title_token_f1(pred.title, gold.title)

    type_sim = 1.0 if pred.type == gold.type else 0.0

    if pred.location is None and gold.location is None:
        loc_sim = 1.0
    elif pred.location is not None and gold.location is not None:
        loc_sim = fuzz.ratio(pred.location.lower(), gold.location.lower()) / 100.0
    else:
        loc_sim = 0.0

    rest = (
        weights.w_time * time_sim
        + weights.w_title * title_sim
        + weights.w_type * type_sim
        + weights.w_loc * loc_sim
    )
    gate = 1.0 if date_ok else weights.wrong_date_factor
    return weights.w_date * float(date_ok) + gate * rest


def compute_hungarian_match_score(
    preds: list[Event],
    golds: list[Event],
    weights: RewardWeights,
) -> tuple[float, int]:
    """Compute normalized bipartite match score using the Hungarian algorithm.

    Returns:
        (r_match, num_matched)
    """
    n_pred = len(preds)
    n_gold = len(golds)

    if n_pred == 0 or n_gold == 0:
        return 0.0, 0

    sim_matrix = np.zeros((n_pred, n_gold), dtype=np.float64)
    for i, p in enumerate(preds):
        for j, g in enumerate(golds):
            sim_matrix[i, j] = _event_similarity(p, g, weights)

    row_ind, col_ind = linear_sum_assignment(1.0 - sim_matrix)

    matched_sim = float(sim_matrix[row_ind, col_ind].sum())
    num_matched = int((sim_matrix[row_ind, col_ind] >= 0.5).sum())

    # Dividing by max(|pred|, |gold|) penalizes both hallucinated and missed events
    r_match = matched_sim / max(n_pred, n_gold)
    return r_match, num_matched


def validate_gold(raw_gold: Any) -> list[Event]:
    """Coerce a gold label (list of dicts / Events, or JSON string) into list[Event].

    Raises ValueError on anything malformed: a silently dropped gold event would
    turn into a wrong "this email has no events" reward signal.
    """
    if isinstance(raw_gold, str):
        parsed = parse_completion(raw_gold)
        if parsed is None:
            raise ValueError(f"Gold label is not a valid event JSON list: {raw_gold!r}")
        return parsed
    if not isinstance(raw_gold, list):
        raise ValueError(f"Gold label must be a list, got {type(raw_gold).__name__}")
    out = []
    for item in raw_gold:
        if isinstance(item, Event):
            out.append(item)
        elif isinstance(item, dict):
            out.append(Event.model_validate(item))
        else:
            raise ValueError(f"Gold event must be a dict, got {type(item).__name__}")
    return out


def completion_text(completion: Any) -> str:
    """Return the text of a completion, accepting TRL's conversational format."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[-1], dict):
        return str(completion[-1].get("content", ""))
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    return str(completion)


def has_text_outside_json(raw: str) -> bool:
    """True if the completion contains anything besides the JSON array (an optional ```json fence is fine)."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    payload = extract_json_payload(text)
    return text.strip() != payload.strip()


def compute_reward_single(
    completion: Any,
    gold: Union[list[dict], list[Event], str],
    prompt: Optional[str] = None,
    weights: Optional[RewardWeights] = None,
) -> RewardBreakdown:
    """Compute the detailed reward breakdown for a single model completion."""
    if weights is None:
        weights = RewardWeights()

    golds = validate_gold(gold)
    raw = completion_text(completion).strip()
    preds = parse_completion(raw)
    n_gold = len(golds)

    if preds is None:
        # Invalid JSON / schema: no match credit at all
        r_format = -1.0
        r_match = 0.0
        r_empty = -1.0 if n_gold == 0 else -0.5
        r_len = -1.0
        total = weights.w_format * r_format + weights.w_empty * r_empty + weights.w_len * r_len
        return RewardBreakdown(total, r_format, r_match, r_empty, r_len, 0, n_gold, 0)

    r_format = 1.0
    n_pred = len(preds)
    r_match, num_matched = 0.0, 0

    if n_gold == 0:
        # +1 for correctly answering [], -1 for hallucinating events
        r_empty = 1.0 if n_pred == 0 else -1.0
    elif n_pred == 0:
        # Missing every event is penalized so that a constant [] policy is not safe
        r_empty = -0.5
    else:
        r_empty = 0.0
        r_match, num_matched = compute_hungarian_match_score(preds, golds, weights)

    r_len = -1.0 if has_text_outside_json(raw) else 0.0
    if len(raw) > weights.max_chars:
        r_len -= 0.5

    total = (
        weights.w_format * r_format
        + weights.w_match * r_match
        + weights.w_empty * r_empty
        + weights.w_len * r_len
    )
    return RewardBreakdown(total, r_format, r_match, r_empty, r_len, n_pred, n_gold, num_matched)


# ==============================================================================
# TRL GRPOTrainer compatible reward functions.
# Only reward_func_total drives training (see reward_weights in train_grpo.py);
# the component functions are logged for analysis.
# ==============================================================================

def _breakdowns(completions: Sequence[Any], kwargs: dict[str, Any]) -> list[RewardBreakdown]:
    gold_list = kwargs["gold"]
    return [compute_reward_single(c, g) for c, g in zip(completions, gold_list)]


def reward_func_total(prompts: Sequence[Any], completions: Sequence[Any], **kwargs: Any) -> list[float]:
    """Overall weighted reward R."""
    return [b.total for b in _breakdowns(completions, kwargs)]


def reward_func_format(prompts: Sequence[Any], completions: Sequence[Any], **kwargs: Any) -> list[float]:
    """Format validity reward (+1 for a valid JSON list of Events, -1 otherwise)."""
    return [1.0 if parse_completion(completion_text(c)) is not None else -1.0 for c in completions]


def reward_func_match(prompts: Sequence[Any], completions: Sequence[Any], **kwargs: Any) -> list[float]:
    """Hungarian matching reward component."""
    return [b.r_match for b in _breakdowns(completions, kwargs)]


def reward_func_empty(prompts: Sequence[Any], completions: Sequence[Any], **kwargs: Any) -> list[float]:
    """Empty precision/recall reward component."""
    return [b.r_empty for b in _breakdowns(completions, kwargs)]


def reward_func_len(prompts: Sequence[Any], completions: Sequence[Any], **kwargs: Any) -> list[float]:
    """Cleanliness / length penalty component."""
    return [b.r_len for b in _breakdowns(completions, kwargs)]
