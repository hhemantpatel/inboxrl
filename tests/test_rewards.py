"""Tests for reward computation and exploit guards."""

from inboxrl.rewards import compute_reward_single, RewardWeights
from inboxrl.schema import Event


def test_perfect_reward():
    gold = [
        {
            "title": "CSL3090 Assignment 2",
            "type": "deadline",
            "date": "2026-11-05",
            "start_time": "23:59",
            "end_time": None,
            "location": None,
        }
    ]
    completion = '[{"title": "CSL3090 Assignment 2", "type": "deadline", "date": "2026-11-05", "start_time": "23:59", "end_time": null, "location": null}]'
    res = compute_reward_single(completion, gold)
    assert res.r_format == 1.0
    assert res.r_match >= 0.95
    assert res.total >= 2.4


def test_empty_both_reward():
    gold = []
    completion = "[]"
    res = compute_reward_single(completion, gold)
    assert res.r_format == 1.0
    assert res.r_empty == 1.0
    assert res.total >= 1.4


def test_empty_gold_with_hallucination():
    gold = []
    completion = '[{"title": "Invented Event", "type": "event", "date": "2026-11-01", "start_time": null, "end_time": null, "location": null}]'
    res = compute_reward_single(completion, gold)
    assert res.r_format == 1.0
    assert res.r_empty == -1.0
    # Hallucinating when gold is empty must give negative or zero total reward
    assert res.total < 0.0


def test_constant_empty_exploit_punished():
    """Verify that returning [] when events exist is penalized."""
    gold = [
        {
            "title": "Exam",
            "type": "exam",
            "date": "2026-10-10",
            "start_time": "09:00",
            "location": "LHC-101",
        }
    ]
    completion = "[]"
    res = compute_reward_single(completion, gold)
    # r_empty is -0.5, r_match is 0.0
    assert res.r_empty == -0.5
    assert res.r_match == 0.0
    assert res.total <= 0.0


def test_invalid_json_penalty():
    gold = [
        {
            "title": "Exam",
            "type": "exam",
            "date": "2026-10-10",
            "start_time": "09:00",
            "location": "LHC-101",
        }
    ]
    completion = "I found an exam on 10 Oct 2026."
    res = compute_reward_single(completion, gold)
    assert res.r_format == -1.0
    assert res.total < 0.0


def test_hungarian_order_invariance():
    """Matching should be invariant to the order of events in the JSON list."""
    gold = [
        {
            "title": "Review Session",
            "type": "meeting",
            "date": "2026-10-15",
            "start_time": "16:00",
            "location": "LHC-101",
        },
        {
            "title": "Midsem Exam",
            "type": "exam",
            "date": "2026-10-20",
            "start_time": "10:00",
            "location": "LHC-102",
        },
    ]
    # Prediction has them in reversed order
    completion = """[
        {"title": "Midsem Exam", "type": "exam", "date": "2026-10-20", "start_time": "10:00", "end_time": null, "location": "LHC-102"},
        {"title": "Review Session", "type": "meeting", "date": "2026-10-15", "start_time": "16:00", "end_time": null, "location": "LHC-101"}
    ]"""
    res = compute_reward_single(completion, gold)
    assert res.r_match >= 0.95
    assert res.num_matched == 2


# ---------------------------------------------------------------------------
# Reward-hacking guards
# ---------------------------------------------------------------------------
import json

import pytest

from inboxrl.rewards import has_text_outside_json, reward_func_total, title_token_f1

GOLD_ONE = [
    {
        "title": "CSL3090 Assignment Submission",
        "type": "deadline",
        "date": "2026-11-05",
        "start_time": "23:59",
        "end_time": None,
        "location": None,
    }
]


def _pred(**overrides):
    ev = dict(GOLD_ONE[0])
    ev.update(overrides)
    return json.dumps([ev])


def test_short_title_exploit_is_not_worth_it():
    perfect = compute_reward_single(_pred(), GOLD_ONE).total
    one_letter = compute_reward_single(_pred(title="a"), GOLD_ONE).total
    subset = compute_reward_single(_pred(title="Assignment"), GOLD_ONE).total
    assert perfect == pytest.approx(2.5)
    assert one_letter <= perfect - 0.45
    assert one_letter < subset < perfect


def test_wrong_date_loses_most_of_the_reward():
    wrong_date = compute_reward_single(_pred(date="2026-01-01"), GOLD_ONE).total
    assert wrong_date < 1.0
    # ...but still beats giving up with [] so early exploration gets signal
    assert wrong_date > compute_reward_single("[]", GOLD_ONE).total


def test_prose_around_json_costs_at_least_quarter_point():
    clean = compute_reward_single(_pred(), GOLD_ONE).total
    chatty = compute_reward_single("Sure! Here you go: " + _pred() + " Hope it helps.", GOLD_ONE).total
    assert clean - chatty >= 0.25


def test_code_fence_is_not_penalized():
    fenced = "```json\n" + _pred() + "\n```"
    assert not has_text_outside_json(fenced)
    assert compute_reward_single(fenced, GOLD_ONE).total == pytest.approx(2.5)


def test_invalid_gold_raises():
    with pytest.raises(ValueError):
        compute_reward_single("[]", [{"title": "x", "type": "party", "date": "2026-13-01"}])


def test_conversational_completions_are_accepted():
    comps = [[{"role": "assistant", "content": _pred()}]]
    assert reward_func_total(["p"], comps, gold=[GOLD_ONE]) == [pytest.approx(2.5)]


def test_title_token_f1():
    assert title_token_f1("CSL3090 Quiz 2", "CSL3090 Quiz 2") == 1.0
    assert title_token_f1("", "CSL3090 Quiz 2") == 0.0
    assert 0.0 < title_token_f1("Quiz", "CSL3090 Quiz 2") < 0.6


def test_exploit_policies_lose_to_a_decent_policy_on_synthetic_data():
    """Constant [] and 'copy every date' policies must score below an imperfect but honest policy."""
    from inboxrl.data.synth import generate_dataset

    data = generate_dataset(200, seed=7, split="train")

    def mean_reward(policy):
        return sum(compute_reward_single(policy(r), r["gold"]).total for r in data) / len(data)

    def constant_empty(r):
        return "[]"

    def every_date(r):
        # One event per distinct date mentioned anywhere (received date + all gold dates + a decoy)
        dates = {r["received_date"]} | {g["date"] for g in r["gold"]}
        return json.dumps([{"title": "Event", "type": "event", "date": d} for d in sorted(dates)])

    def decent(r):
        # Right dates, generic titles, no times or venues
        return json.dumps([{"title": g["title"].split()[0], "type": g["type"], "date": g["date"]} for g in r["gold"]])

    assert mean_reward(constant_empty) < mean_reward(decent)
    assert mean_reward(every_date) < mean_reward(decent)


# ---------------------------------------------------------------- date verifier (v2 reward) ----

def test_date_closeness_default_is_all_or_nothing():
    exact = compute_reward_single(_pred(), GOLD_ONE).total
    off1 = compute_reward_single(_pred(date="2026-11-06"), GOLD_ONE).total
    off7 = compute_reward_single(_pred(date="2026-11-12"), GOLD_ONE).total
    assert off1 == pytest.approx(off7)
    assert exact > off1


def test_date_closeness_orders_near_misses_but_never_beats_a_hit():
    w = RewardWeights(date_closeness=0.5)
    exact = compute_reward_single(_pred(), GOLD_ONE, weights=w).total
    off1 = compute_reward_single(_pred(date="2026-11-06"), GOLD_ONE, weights=w).total
    off7 = compute_reward_single(_pred(date="2026-11-12"), GOLD_ONE, weights=w).total
    far = compute_reward_single(_pred(date="2027-01-01"), GOLD_ONE, weights=w).total
    assert exact > off1 > off7 > far
    assert off1 < 0.5 * exact  # a near miss earns well under half of a correct event


def test_make_reward_funcs_matches_module_level_functions():
    from inboxrl.rewards import make_reward_funcs

    funcs = make_reward_funcs(RewardWeights())
    comps = [_pred(), "[]", "not json"]
    gold = [GOLD_ONE] * 3
    assert funcs[0](["p"] * 3, comps, gold=gold) == reward_func_total(["p"] * 3, comps, gold=gold)
    assert [f.__name__ for f in funcs] == [
        "reward_func_total", "reward_func_format", "reward_func_match", "reward_func_empty", "reward_func_len",
    ]
