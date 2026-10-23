"""Schema definition and parser for calendar events extracted from emails."""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, field_validator

EventType = Literal["deadline", "exam", "class", "event", "meeting"]

ALLOWED_EVENT_TYPES = {"deadline", "exam", "class", "event", "meeting"}

TYPE_SYNONYMS = {
    # Deadline synonyms
    "assignment": "deadline",
    "submission": "deadline",
    "due": "deadline",
    "due date": "deadline",
    "project": "deadline",
    "homework": "deadline",
    # Exam synonyms
    "quiz": "exam",
    "midsem": "exam",
    "endsem": "exam",
    "test": "exam",
    "midterm": "exam",
    "final": "exam",
    "evaluation": "exam",
    # Class synonyms
    "lecture": "class",
    "lab": "class",
    "tutorial": "class",
    "session": "class",
    # Event synonyms
    "talk": "event",
    "seminar": "event",
    "workshop": "event",
    "webinar": "event",
    "conference": "event",
    "fest": "event",
    "hackathon": "event",
    # Meeting synonyms
    "discussion": "meeting",
    "sync": "meeting",
    "office hours": "meeting",
    "interview": "meeting",
}


def normalize_event_type(val: str) -> str:
    cleaned = val.strip().lower()
    if cleaned in ALLOWED_EVENT_TYPES:
        return cleaned
    if cleaned in TYPE_SYNONYMS:
        return TYPE_SYNONYMS[cleaned]
    return cleaned


def normalize_time_str(val: Optional[str]) -> Optional[str]:
    if val is None:
        return None
    val = val.strip().lower()
    if not val or val == "null" or val == "none":
        return None
    if val in {"midnight", "eod", "end of day", "24:00"}:
        return "23:59"
    if val in {"noon", "midday"}:
        return "12:00"
    
    # Try parsing 24-hour HH:MM or HH:MM:SS
    m = re.match(r"^(\d{1,2}):(\d{2})(?::\d{2})?$", val)
    if m:
        h, mn = int(m.group(1)), int(m.group(2))
        if 0 <= h <= 23 and 0 <= mn <= 59:
            return f"{h:02d}:{mn:02d}"
        if h == 24 and mn == 0:
            return "23:59"

    # Try parsing 12-hour AM/PM e.g. 5 pm, 5:30pm, 11:59 pm
    m_ampm = re.match(r"^(\d{1,2})(?::(\d{2}))?\s*([ap]m)$", val)
    if m_ampm:
        h = int(m_ampm.group(1))
        mn = int(m_ampm.group(2)) if m_ampm.group(2) else 0
        meridiem = m_ampm.group(3)
        if 1 <= h <= 12 and 0 <= mn <= 59:
            if meridiem == "pm" and h < 12:
                h += 12
            elif meridiem == "am" and h == 12:
                h = 0
            return f"{h:02d}:{mn:02d}"

    return val


class Event(BaseModel):
    title: str = Field(..., description="Clear title/summary of the event")
    type: EventType = Field(..., description="deadline | exam | class | event | meeting")
    date: str = Field(..., description="Event date in YYYY-MM-DD format")
    start_time: Optional[str] = Field(None, description="Start time in HH:MM format (24-hour)")
    end_time: Optional[str] = Field(None, description="End time in HH:MM format (24-hour)")
    location: Optional[str] = Field(None, description="Venue or platform or null")

    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("Title cannot be empty")
        return s

    @field_validator("type", mode="before")
    @classmethod
    def validate_type(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise ValueError("Type must be a string")
        norm = normalize_event_type(v)
        if norm not in ALLOWED_EVENT_TYPES:
            raise ValueError(f"Type '{v}' is not one of {sorted(ALLOWED_EVENT_TYPES)}")
        return norm

    @field_validator("date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        s = v.strip()
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", s):
            raise ValueError(f"Date '{s}' must be in YYYY-MM-DD format")
        try:
            datetime.strptime(s, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(f"Invalid calendar date '{s}': {e}")
        return s

    @field_validator("start_time", "end_time", mode="before")
    @classmethod
    def validate_time(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        norm = normalize_time_str(str(v))
        if norm is None:
            return None
        if not re.match(r"^(?:[01]\d|2[0-3]):[0-5]\d$", norm):
            raise ValueError(f"Time '{v}' cannot be normalized to valid 24-hr HH:MM format")
        return norm

    @field_validator("location")
    @classmethod
    def validate_location(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip()
        if not s or s.lower() in {"null", "none", "n/a", "tba"}:
            return None
        return s


def extract_json_payload(raw_text: str) -> str:
    """Extract json array string from completion, stripping code fences or whitespace."""
    text = raw_text.strip()

    # If wrapped in markdown code fence ```json ... ```
    match = re.search(r"```(?:json)?\s*(\[\s*.*?\])\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Search for top-level JSON array [...]
    # Find first '[' and matching last ']'
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1].strip()

    return text


def parse_completion(raw_text: str) -> Optional[list[Event]]:
    """Parse model completion into a list of validated Event objects.
    
    Returns:
        list[Event] if valid JSON array matching Event schema.
        None if parsing or validation fails.
    """
    if not raw_text or not raw_text.strip():
        return None

    cleaned = extract_json_payload(raw_text)
    try:
        data = json.loads(cleaned)
    except Exception:
        return None

    if not isinstance(data, list):
        return None

    events: list[Event] = []
    for item in data:
        if not isinstance(item, dict):
            return None
        try:
            event = Event.model_validate(item)
            events.append(event)
        except Exception:
            return None

    return events
