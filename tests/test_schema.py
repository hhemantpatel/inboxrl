"""Tests for Event schema validation and completion parsing."""

import pytest
from inboxrl.schema import Event, normalize_event_type, normalize_time_str, parse_completion


def test_normalize_event_type():
    assert normalize_event_type("quiz") == "exam"
    assert normalize_event_type("assignment") == "deadline"
    assert normalize_event_type("lecture") == "class"
    assert normalize_event_type("talk") == "event"
    assert normalize_event_type("sync") == "meeting"
    assert normalize_event_type("DEADLINE") == "deadline"


def test_normalize_time():
    assert normalize_time_str("midnight") == "23:59"
    assert normalize_time_str("noon") == "12:00"
    assert normalize_time_str("5 pm") == "17:00"
    assert normalize_time_str("5:30pm") == "17:30"
    assert normalize_time_str("10:00") == "10:00"
    assert normalize_time_str("24:00") == "23:59"
    assert normalize_time_str(None) is None
    assert normalize_time_str("null") is None


def test_event_model_valid():
    ev = Event(
        title="CSL3090 Assignment 2",
        type="assignment",  # normalizes to deadline
        date="2026-11-05",
        start_time="midnight",  # normalizes to 23:59
        location="Online",
    )
    assert ev.type == "deadline"
    assert ev.start_time == "23:59"
    assert ev.date == "2026-11-05"


def test_event_model_invalid_date():
    with pytest.raises(ValueError):
        Event(title="Test", type="deadline", date="2026-02-30")


def test_parse_completion_clean_json():
    raw = '[{"title": "Quiz 1", "type": "exam", "date": "2026-09-15", "start_time": "10:00", "end_time": null, "location": "LHC-101"}]'
    events = parse_completion(raw)
    assert events is not None
    assert len(events) == 1
    assert events[0].title == "Quiz 1"
    assert events[0].type == "exam"


def test_parse_completion_markdown_fence():
    raw = """Here is the result:
```json
[
  {
    "title": "Guest Lecture",
    "type": "event",
    "date": "2026-10-20",
    "start_time": "15:00",
    "end_time": null,
    "location": "Auditorium"
  }
]
```
Hope this helps!"""
    events = parse_completion(raw)
    assert events is not None
    assert len(events) == 1
    assert events[0].title == "Guest Lecture"


def test_parse_completion_empty():
    assert parse_completion("[]") == []
    assert parse_completion("```json\n[]\n```") == []


def test_parse_completion_invalid():
    assert parse_completion("Not a json at all") is None
    assert parse_completion('{"error": "no event"}') is None
