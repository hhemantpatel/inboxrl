"""End-to-end CLI: email text or mbox archive -> calendar events -> .ics file.

    python -m inboxrl.demo.extract --text "Quiz 2 of CSL3090 is on 5 Nov at 10 am in LHC-101" --received-date 2026-10-23
    python -m inboxrl.demo.extract --mbox ~/Takeout/Mail/Inbox.mbox --since 2026-10-01 --out events.ics
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Optional

from inboxrl.agent import DEFAULT_BASE_MODEL, generate_batch, load_model
from inboxrl.demo.ics import save_ics_file
from inboxrl.evaluate import adapter_settings
from inboxrl.prompts import build_prompt
from inboxrl.schema import Event, parse_completion


def extract_events(
    model: Any, tokenizer: Any, emails: list[dict[str, Any]], batch_size: int = 8, prompt_version: int = 1,
    max_prompt_tokens: int = 640, max_new_tokens: int = 192,
) -> list[list[Event]]:
    """Run the policy on email records ({received_date, sender, subject, body}); invalid outputs give [].

    The prompt version and token limits must be the ones the adapter was trained with (see `adapter_settings`).
    """
    prompts = [build_prompt(tokenizer, e["received_date"], e["sender"], e["subject"], e["body"], max_prompt_tokens,
                            prompt_version) for e in emails]
    completions = generate_batch(model, tokenizer, prompts, max_new_tokens=max_new_tokens, batch_size=batch_size)
    return [parse_completion(c) or [] for c in completions]


def dedupe(events: list[Event]) -> list[Event]:
    """Drop repeats of the same event (e.g. a reminder email for an already-announced quiz)."""
    seen: set[tuple[str, str, Optional[str]]] = set()
    out = []
    for ev in events:
        key = (ev.title.strip().lower(), ev.date, ev.start_time)
        if key not in seen:
            seen.add(key)
            out.append(ev)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="InboxRL: extract calendar events from emails into an .ics file")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", type=str, help="Raw email body to process")
    src.add_argument("--mbox", type=str, help="Path to an mbox archive (e.g. from Google Takeout)")
    parser.add_argument("--subject", type=str, default="(no subject)")
    parser.add_argument("--sender", type=str, default="unknown@iitj.ac.in")
    parser.add_argument("--received-date", type=str, default=None, help="YYYY-MM-DD (default: today)")
    parser.add_argument("--since", type=str, default=None, help="mbox: only emails received on/after YYYY-MM-DD")
    parser.add_argument("--sender-filter", type=str, default="@iitj.ac.in", help="mbox: keep senders containing this")
    parser.add_argument("--model-path", type=str, default="checkpoints/final", help="Trained LoRA adapter directory")
    parser.add_argument("--base-only", action="store_true", help="Use the untrained base model (for comparison)")
    parser.add_argument("--device", type=str, default=None, help="cpu / cuda (default: auto)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out", type=str, default="events.ics")
    args = parser.parse_args()

    if args.text:
        emails = [{
            "received_date": args.received_date or date.today().isoformat(),
            "sender": args.sender,
            "subject": args.subject,
            "body": args.text,
        }]
    else:
        from inboxrl.data.ingest_mbox import parse_mbox_file

        emails = parse_mbox_file(Path(args.mbox), sender_filter=args.sender_filter, anonymize=False)
        if args.since:
            emails = [e for e in emails if e["received_date"] >= args.since]
        print(f"Processing {len(emails)} emails from {args.mbox}...")
        if not emails:
            sys.exit("No matching emails.")

    print("Loading model...")
    if args.base_only:
        model, tok = load_model(base_model_name=DEFAULT_BASE_MODEL, device=args.device)
    else:
        model, tok = load_model(adapter_path=args.model_path, device=args.device)

    settings = {} if args.base_only else adapter_settings(args.model_path)
    per_email = extract_events(model, tok, emails, batch_size=args.batch_size, **settings)
    all_events = dedupe([ev for evs in per_email for ev in evs])

    if args.text:
        print(json.dumps([ev.model_dump() for ev in all_events], indent=2))
    if all_events:
        save_ics_file(all_events, Path(args.out))
    else:
        print("No events detected.")


if __name__ == "__main__":
    main()
