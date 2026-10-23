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
