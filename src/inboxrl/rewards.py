"""Reward functions and Hungarian bipartite matching for InboxRL.

Problem Formulation & Permutation Invariance:
    Extracting calendar events from unstructured email text requires emitting a JSON
    array of structured Event schemas: [Event_1, Event_2, ...]. Because a predicted list
    is an unordered set of events, index-by-index comparison (e.g. matching pred[0] to gold[0])
    would artificially penalize valid permutations (e.g., [e_B, e_A] vs gold [e_A, e_B]).

    To achieve permutation invariance, InboxRL frames event alignment as a Maximum Weight
    Bipartite Matching problem between predicted events P = {p_1, ..., p_m} and ground-truth
    events G = {g_1, ..., g_n}. We solve for the optimal bijection:
        pi* = argmax_{pi} sum_{(i, j) in pi} S(p_i, g_j)
    using the Kuhn-Munkres (Hungarian) algorithm in O(max(m, n)^3) time.

Reward Decomposition:
    The overall scalar reward R is a weighted sum of four orthogonal signals:
        R_total = w_format * R_format + w_match * R_match + w_empty * R_empty + w_len * R_len

    1. Format Validity (R_format):
       +1.0 if the output strictly parses into valid Event JSON schemas; -1.0 otherwise.
    2. Hungarian Match (R_match):
       Normalized bipartite similarity: sum_{(i,j) in pi*} S(p_i, g_j) / max(|P|, |G|).
       Dividing by max(|P|, |G|) simultaneously penalizes false positives (hallucinations)
       and false negatives (missed events).
    3. Empty Event Precision/Recall (R_empty):
       +1.0 for correctly emitting [] on non-event emails, -1.0 for hallucinating events on
       empty emails, and -0.5 for emitting [] when events were actually present.
    4. Length & Cleanliness Penalty (R_len):
       -1.0 if conversational fluff exists outside the JSON payload; -0.5 if length exceeds max_chars.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date as _date
from typing import Any, Optional, Sequence, Union

import numpy as np
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment

from inboxrl.schema import Event, extract_json_payload, parse_completion


@dataclass
class RewardWeights:
    """Hyperparameters and sub-field weights for the composite RL reward function.

    Attributes:
        w_format: Weight for JSON syntactic validity (+1.0 / -1.0).
        w_match: Weight for bipartite Hungarian event matching.
        w_empty: Weight for empty-email precision/recall reward.
        w_len: Weight for brevity and zero-fluff text penalty.
        w_date: Relative weight for ISO date agreement (YYYY-MM-DD).
        w_time: Relative weight for start time match (HH:MM or None).
        w_title: Relative weight for title token-level F1 overlap.
        w_type: Relative weight for event category agreement.
        w_loc: Relative weight for fuzzy location string similarity.
        wrong_date_factor: Gating factor applied to non-date similarities when date is wrong.
            Because an event scheduled on the wrong date is calendar-useless, wrong dates
            cap non-date field credit to wrong_date_factor * sum(non-date weights).
        max_chars: Character limit before length penalty kicks in.
        date_closeness: Partial credit ceiling for near-miss dates (linear decay).
        date_horizon_days: Maximum date deviation (days) eligible for partial credit.
    """
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
    # Date verifier: a wrong date earns up to `date_closeness` of the date credit, linearly decaying to 0 at
    # `date_horizon_days` off. An exact date still earns 1.0, and the other fields stay gated on the exact date,
    # so a near miss never beats a hit. 0 (default) keeps the original all-or-nothing date reward.
    date_closeness: float = 0.0
    date_horizon_days: int = 14


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


def date_credit(pred_date: str, gold_date: str, weights: RewardWeights) -> float:
    """Credit in [0, 1] for a predicted date: 1 if exact, else a distance-decayed fraction of `date_closeness`.

    A calendar-arithmetic verifier: off-by-one-day and off-by-one-week mistakes ("this Friday" vs "next Friday")
    are told apart from random dates, which makes the signal dense in groups where no completion is exactly right.
    """
    if pred_date == gold_date:
        return 1.0
    if weights.date_closeness <= 0.0:
        return 0.0
    delta = abs((_date.fromisoformat(pred_date) - _date.fromisoformat(gold_date)).days)
    return weights.date_closeness * max(0.0, 1.0 - delta / weights.date_horizon_days)


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
    return weights.w_date * date_credit(pred.date, gold.date, weights) + gate * rest


def compute_hungarian_match_score(
    preds: list[Event],
    golds: list[Event],
    weights: RewardWeights,
) -> tuple[float, int]:
    """Compute normalized bipartite match score using the Hungarian algorithm.

    Mathematical Formulation:
        Let P = {p_1, ..., p_m} be predicted events and G = {g_1, ..., g_n} be gold events.
        1. Construct similarity matrix S in [0, 1]^{m x n} where S_{i, j} = _event_similarity(p_i, g_j).
        2. Convert to cost matrix C_{i, j} = 1.0 - S_{i, j} for minimum weight assignment.
        3. Solve Kuhn-Munkres: (row_ind, col_ind) = linear_sum_assignment(C).
        4. Normalize total matched similarity by max(m, n):
               r_match = (sum_{k} S_{row[k], col[k]}) / max(m, n)
           This normalization strictly bounds r_match in [0, 1] while penalizing both
           hallucinated events (when m > n) and omitted events (when m < n).

    Returns:
        tuple[float, int]:
            - r_match: Normalized match reward in [0.0, 1.0].
            - num_matched: Count of matched pairs with high semantic similarity (S >= 0.5).
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

def _breakdowns(
    completions: Sequence[Any], kwargs: dict[str, Any], weights: Optional[RewardWeights] = None
) -> list[RewardBreakdown]:
    gold_list = kwargs["gold"]
    return [compute_reward_single(c, g, weights=weights) for c, g in zip(completions, gold_list)]


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


def make_reward_funcs(weights: RewardWeights) -> list:
    """The five TRL reward functions (total, format, match, empty, len) bound to custom RewardWeights."""

    def reward_func_total(prompts, completions, **kwargs):
        return [b.total for b in _breakdowns(completions, kwargs, weights)]

    def reward_func_format(prompts, completions, **kwargs):
        return [1.0 if parse_completion(completion_text(c)) is not None else -1.0 for c in completions]

    def reward_func_match(prompts, completions, **kwargs):
        return [b.r_match for b in _breakdowns(completions, kwargs, weights)]

    def reward_func_empty(prompts, completions, **kwargs):
        return [b.r_empty for b in _breakdowns(completions, kwargs, weights)]

    def reward_func_len(prompts, completions, **kwargs):
        return [b.r_len for b in _breakdowns(completions, kwargs, weights)]

    return [reward_func_total, reward_func_format, reward_func_match, reward_func_empty, reward_func_len]
