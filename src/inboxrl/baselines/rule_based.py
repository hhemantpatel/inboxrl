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
        
        # Check if email is explicitly a non-event notification
        no_event_cues = ["marks published", "grades published", "maintenance completed", "feedback", "reading material"]
        if any(cue in full_text.lower() for cue in no_event_cues) and "deadline" not in full_text.lower():
            # Likely no event
            return []

        # Reference date for relative parsing
        try:
            ref_dt = datetime.strptime(received_date_str, "%Y-%m-%d")
        except Exception:
            ref_dt = datetime.now()

        # Classify event type
        detected_type = "event"
        for etype, keywords in self.type_keywords.items():
            if any(kw in full_text.lower() for kw in keywords):
                detected_type = etype
                break

        # Search for venue
        loc_match = self.venue_pattern.search(full_text)
        location = loc_match.group(1).strip() if loc_match else None

        # Search for time
        time_match = self.time_pattern.search(full_text)
        start_time = normalize_time_str(time_match.group(1)) if time_match else None

        # Detect candidate dates using regex expressions
        # Patterns: DD Month YYYY, DD/MM/YYYY, tomorrow, next [Day], this [Day]
        date_candidates = re.findall(
            r"\b(?:\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*(?:\d{4})?|\d{1,2}/\d{1,2}/\d{4}|tomorrow|day after tomorrow|this (?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)|next (?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday))\b",
            full_text,
            re.IGNORECASE,
        )

        if not date_candidates:
            return []

        # Take the most prominent date (often the last mentioned in case of extensions)
        best_date_str = None
        for candidate in reversed(date_candidates):
            parsed = dateparser.parse(
                candidate,
                settings={"RELATIVE_BASE": ref_dt, "PREFER_DATES_FROM": "future"},
            )
            if parsed:
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
