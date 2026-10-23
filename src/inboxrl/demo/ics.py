"""Export extracted events to iCalendar (.ics) format for Google/Apple Calendar."""

from __future__ import annotations

import uuid
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Sequence

from icalendar import Calendar, Event as ICalEvent

from inboxrl.schema import Event


def events_to_ics(events: Sequence[Event], calendar_name: str = "InboxRL Calendar") -> bytes:
    """Convert a sequence of Event objects into an RFC 5545 iCalendar byte stream."""
    cal = Calendar()
    cal.add("prodid", "-//InboxRL AI Event Extractor//EN")
    cal.add("version", "2.0")
    cal.add("x-wr-calname", calendar_name)

    for ev in events:
        try:
            ev_date = datetime.strptime(ev.date, "%Y-%m-%d").date()
        except Exception:
            continue

        ical_ev = ICalEvent()
        ical_ev.add("uid", f"{uuid.uuid4()}@inboxrl.ai")
        ical_ev.add("dtstamp", datetime.utcnow())
        ical_ev.add("summary", ev.title)
        ical_ev.add("categories", [ev.type.upper()])

        if ev.location:
            ical_ev.add("location", ev.location)

        if ev.start_time:
            try:
                st_h, st_m = map(int, ev.start_time.split(":"))
                start_dt = datetime.combine(ev_date, time(st_h, st_m))
                ical_ev.add("dtstart", start_dt)

                if ev.end_time:
                    et_h, et_m = map(int, ev.end_time.split(":"))
                    end_dt = datetime.combine(ev_date, time(et_h, et_m))
                    ical_ev.add("dtend", end_dt)
                else:
                    # Default 1-hour duration for scheduled meetings/events, 0 for deadlines
                    if ev.type == "deadline":
                        ical_ev.add("dtend", start_dt)
                    else:
                        ical_ev.add("dtend", start_dt + timedelta(hours=1))
            except Exception:
                ical_ev.add("dtstart", ev_date)
        else:
            # All-day event
            ical_ev.add("dtstart", ev_date)
            ical_ev.add("dtend", ev_date + timedelta(days=1))

        cal.add_component(ical_ev)

    return cal.to_ical()


def save_ics_file(events: Sequence[Event], out_path: Path, calendar_name: str = "InboxRL") -> None:
    """Write events to an .ics file on disk."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    content = events_to_ics(events, calendar_name=calendar_name)
    with open(out_path, "wb") as f:
        f.write(content)
    print(f"Exported {len(events)} event(s) to {out_path}")
