"""Tests for synthetic email dataset generation and difficulty levels."""

from collections import Counter
from datetime import date, datetime

import pytest

from inboxrl.data.synth import generate_dataset, resolve_relative_date
from inboxrl.rewards import validate_gold


@pytest.fixture(scope="module")
def train():
    return generate_dataset(500, seed=1, split="train")


@pytest.fixture(scope="module")
def test_split():
    return generate_dataset(300, seed=1, split="test")


def test_structure_and_gold_validity(train):
    for s in train:
        assert {"id", "received_date", "sender", "subject", "body", "gold", "level", "template_ids"} <= s.keys()
        assert s["level"] in {1, 2, 3, 4, 5}
        received = datetime.strptime(s["received_date"], "%Y-%m-%d").date()
        for ev in validate_gold(s["gold"]):
            assert date.fromisoformat(ev.date) >= received


def test_levels_balanced_and_some_empty(train):
    levels = Counter(s["level"] for s in train)
    assert set(levels.values()) == {100}
    n_empty = sum(1 for s in train if not s["gold"])
    assert 0.08 * len(train) < n_empty < 0.25 * len(train)
    assert any(len(s["gold"]) >= 2 for s in train), "L4 digests should produce multi-event emails"


def test_gold_location_appears_in_body(train, test_split):
    for s in train + test_split:
        for ev in s["gold"]:
            if ev["location"] is not None:
                assert ev["location"] in s["body"], (s["id"], ev["location"])


def test_no_time_label_inconsistency(train):
    """'end of day' / 'EOD' / 'midnight' phrasing must always be labelled 23:59."""
    for s in train:
        body = s["body"].lower()
        if any(p in body for p in ("end of day", "eod", "midnight")) and len(s["gold"]) == 1:
            assert s["gold"][0]["start_time"] == "23:59", s["id"]


def test_test_templates_disjoint_from_train(train, test_split):
    train_t = {t for s in train for t in s["template_ids"]}
    test_t = {t for s in test_split for t in s["template_ids"]}
    assert train_t.isdisjoint(test_t)
    train_bodies = {s["body"] for s in train}
    assert not any(s["body"] in train_bodies for s in test_split)


def test_reproducibility():
    assert generate_dataset(20, seed=42) == generate_dataset(20, seed=42)
    assert generate_dataset(20, seed=42) != generate_dataset(20, seed=43)


@pytest.mark.parametrize(
    "expr,expected",
    [
        ("today", date(2026, 10, 21)),
        ("tomorrow", date(2026, 10, 22)),
        ("day after tomorrow", date(2026, 10, 23)),
        ("this Friday", date(2026, 10, 23)),       # 2026-10-21 is a Wednesday
        ("this Wednesday", date(2026, 10, 28)),    # same weekday -> one week ahead
        ("next Friday", date(2026, 10, 30)),
        ("next Monday", date(2026, 11, 2)),
    ],
)
def test_relative_date_convention(expr, expected):
    assert resolve_relative_date(date(2026, 10, 21), expr) == expected


def test_course_code_in_title_is_visible_in_email(train, test_split):
    import re

    for s in train + test_split:
        for ev in s["gold"]:
            code = ev["title"].split()[0]
            if re.fullmatch(r"[A-Z]{3}\d{4}", code):
                assert code in s["subject"] + s["body"], (s["id"], code)
