"""Rule-based extractor baseline using regex heuristics and dateparser."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Optional

import dateparser

from inboxrl.schema import Event, normalize_event_type, normalize_time_str


class RuleBasedExtractor:
    """Heuristic baseline that detects dates, times, and event keywords via regex."""

    def __init__(self) -> None:
        self.time_pattern = re.compile(
            r"\b((?:[01]?\d|2[0-3]):[0-5]\d(?:\s*[ap]m)?|(?:1[0-2]|[1-9])\s*[ap]m|midnight|noon|eod)\b",
            re.IGNORECASE,
        )
        self.type_keywords = {
            "deadline": ["assignment", "submission", "submit", "deadline", "due", "homework"],
            "exam": ["exam", "quiz", "midsem", "endsem", "test", "viva", "evaluation"],
            "class": ["lecture", "class", "lab", "tutorial", "compensatory"],
            "event": ["talk", "seminar", "workshop", "webinar", "distinguished lecture", "session"],
            "meeting": ["meeting", "office hours", "doubt", "discussion", "sync"],
        }
        self.venue_pattern = re.compile(
            r"\b(LHC[- ]\d{3}|Seminar Hall \d|Auditorium|Lab \d{3}|Online(?:\s*\(Google Meet\))?)\b",
            re.IGNORECASE,
        )

    def extract_events(
        self,
        received_date_str: str,
        sender: str,
        subject: str,
        body: str,
    ) -> list[Event]:
        """Extract events using rules."""
        full_text = f"{subject}\n{body}"

        try:
            ref_dt = datetime.strptime(received_date_str, "%Y-%m-%d")
        except ValueError:
            return []

        # Classify event type by the first keyword group that appears
        detected_type = "event"
        for etype, keywords in self.type_keywords.items():
            if any(kw in full_text.lower() for kw in keywords):
                detected_type = etype
                break

        loc_match = self.venue_pattern.search(full_text)
        location = loc_match.group(1).strip() if loc_match else None

        time_match = self.time_pattern.search(full_text)
        start_time = normalize_time_str(time_match.group(1)) if time_match else None

        # Candidate dates: absolute formats and simple relative expressions
        month = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
        weekday = r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"
        date_candidates = re.findall(
            rf"\b(?:\d{{1,2}}(?:st|nd|rd|th)?\s+{month}(?:,?\s*\d{{4}})?"
            rf"|{month}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s*\d{{4}})?"
            rf"|\d{{1,2}}/\d{{1,2}}/\d{{4}}|\d{{4}}-\d{{2}}-\d{{2}}"
            rf"|today|tomorrow|day after tomorrow|(?:this|next) {weekday})\b",
            full_text,
            re.IGNORECASE,
        )

        # Keep the last date that is not in the past (later mentions usually override earlier ones,
        # e.g. "postponed from 3 Nov to 7 Nov"); no future date -> no event.
        best_date_str = None
        for candidate in reversed(date_candidates):
            parsed = dateparser.parse(
                candidate,
                settings={"RELATIVE_BASE": ref_dt, "PREFER_DATES_FROM": "future", "DATE_ORDER": "DMY"},
            )
            if parsed and parsed.date() >= ref_dt.date():
                best_date_str = parsed.strftime("%Y-%m-%d")
                break

        if not best_date_str:
            return []

        # Derive title from subject
        clean_subj = re.sub(r"^(?:Re:|Fwd:)\s*", "", subject, flags=re.IGNORECASE).strip()
        title = clean_subj if len(clean_subj) > 5 else f"{detected_type.capitalize()} on {best_date_str}"

        try:
            event = Event(
                title=title,
                type=detected_type,  # type: ignore
                date=best_date_str,
                start_time=start_time,
                location=location,
            )
            return [event]
        except Exception:
            return []
