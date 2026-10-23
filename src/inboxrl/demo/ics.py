"""Export extracted events to iCalendar (.ics) format for Google/Apple Calendar."""

from __future__ import annotations

import hashlib
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Sequence
from zoneinfo import ZoneInfo

from icalendar import Calendar, Event as ICalEvent

from inboxrl.schema import Event

TZ = ZoneInfo("Asia/Kolkata")
DEFAULT_DURATION = {"exam": 2, "class": 1, "event": 1, "meeting": 1}


def _uid(ev: Event) -> str:
    """Stable UID so re-importing the same event updates it instead of duplicating it."""
    key = f"{ev.title.lower()}|{ev.date}|{ev.start_time}"
    return hashlib.sha1(key.encode()).hexdigest()[:16] + "@inboxrl"


def events_to_ics(events: Sequence[Event], calendar_name: str = "InboxRL") -> bytes:
    """Convert events into an RFC 5545 iCalendar byte stream (times in Asia/Kolkata)."""
    cal = Calendar()
    cal.add("prodid", "-//InboxRL//Calendar Event Extractor//EN")
    cal.add("version", "2.0")
    cal.add("x-wr-calname", calendar_name)

    for ev in events:
        ev_date = date.fromisoformat(ev.date)
        ical_ev = ICalEvent()
        ical_ev.add("uid", _uid(ev))
        ical_ev.add("dtstamp", datetime.now(timezone.utc))
        ical_ev.add("summary", ev.title)
        ical_ev.add("categories", [ev.type.upper()])
        if ev.location:
            ical_ev.add("location", ev.location)

        if ev.start_time:
            start = datetime.combine(ev_date, time.fromisoformat(ev.start_time), tzinfo=TZ)
            if ev.end_time and ev.end_time > ev.start_time:
                end = datetime.combine(ev_date, time.fromisoformat(ev.end_time), tzinfo=TZ)
            elif ev.type == "deadline":
                end = start  # deadlines are instants
            else:
                end = start + timedelta(hours=DEFAULT_DURATION[ev.type])
            ical_ev.add("dtstart", start)
            ical_ev.add("dtend", end)
        else:
            ical_ev.add("dtstart", ev_date)  # all-day
            ical_ev.add("dtend", ev_date + timedelta(days=1))
        cal.add_component(ical_ev)

    cal.add_missing_timezones()
    return cal.to_ical()


def save_ics_file(events: Sequence[Event], out_path: Path, calendar_name: str = "InboxRL") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(events_to_ics(events, calendar_name=calendar_name))
    print(f"Exported {len(events)} event(s) to {out_path}")
