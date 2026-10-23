"""Terminal tool for inspecting and labelling calendar events on real institute emails."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from inboxrl.schema import ALLOWED_EVENT_TYPES, Event


def prompt_event(received_date: str) -> Event | None:
    print("\n--- Enter New Event Details (or press Enter on title to finish) ---")
    title = input("Event Title: ").strip()
    if not title:
        return None

    print(f"Allowed types: {', '.join(sorted(ALLOWED_EVENT_TYPES))}")
    etype = input("Event Type [deadline/exam/class/event/meeting] (default: deadline): ").strip().lower()
    if not etype:
        etype = "deadline"

    date_input = input(f"Date [YYYY-MM-DD] (default: {received_date}): ").strip()
    if not date_input:
        date_input = received_date

    time_input = input("Start Time [HH:MM 24hr or e.g. 17:00, 23:59] (Enter for null): ").strip()
    start_time = time_input if time_input else None

    loc_input = input("Location/Venue (Enter for null): ").strip()
    location = loc_input if loc_input else None

    try:
        return Event(
            title=title,
            type=etype,  # type: ignore
            date=date_input,
            start_time=start_time,
            location=location,
        )
    except Exception as e:
        print(f"Validation error: {e}. Try again.")
        return None


def label_emails(input_file: Path, output_file: Path) -> None:
    if not input_file.exists():
        print(f"File not found: {input_file}")
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)
    existing_ids = set()
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    existing_ids.add(item["id"])

    with open(input_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"Total emails: {len(records)} | Already labeled: {len(existing_ids)}")

    for idx, r in enumerate(records):
        if r["id"] in existing_ids:
            continue

        print("=" * 70)
        print(f"Email [{idx+1}/{len(records)}] ID: {r['id']}")
        print(f"Received Date: {r['received_date']}")
        print(f"From:          {r['sender']}")
        print(f"Subject:       {r['subject']}")
        print("-" * 70)
        print(r["body"][:800] + ("\n... [truncated]" if len(r["body"]) > 800 else ""))
        print("=" * 70)

        events: list[dict[str, Any]] = []
        action = input("Has event? [y=add events, n=no event ([]), s=skip, q=quit]: ").strip().lower()

        if action == "q":
            break
        elif action == "s":
            continue
        elif action == "n":
            r["gold"] = []
        elif action == "y":
            while True:
                ev = prompt_event(r["received_date"])
                if ev is None:
                    break
                events.append(ev.model_dump())
                more = input("Add another event in this email? [y/N]: ").strip().lower()
                if more != "y":
                    break
            r["gold"] = events
        else:
            print("Invalid option. Skipping.")
            continue

        with open(output_file, "a", encoding="utf-8") as out_f:
            out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
        existing_ids.add(r["id"])
        print(f"Saved email {r['id']} with {len(r['gold'])} event(s).\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Terminal CLI for labelling email events.")
    parser.add_argument("--input", type=str, default="data/real/raw_anonymized.jsonl")
    parser.add_argument("--out", type=str, default="data/real/labeled.jsonl")
    args = parser.parse_args()

    label_emails(Path(args.input), Path(args.out))


if __name__ == "__main__":
    main()
