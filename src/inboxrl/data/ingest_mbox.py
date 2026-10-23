"""Ingest Google Takeout mbox files and filter to institute emails."""

from __future__ import annotations

import argparse
import email
import json
import mailbox
from email.header import decode_header
from pathlib import Path
from typing import Any, Optional

from zoneinfo import ZoneInfo

from dateutil import parser as date_parser

from inboxrl.data.anonymize import anonymize_record, load_names

IST = ZoneInfo("Asia/Kolkata")


def decode_str(header_val: Any) -> str:
    """Decode encoded RFC2822 email headers."""
    if not header_val:
        return ""
    decoded_parts = decode_header(header_val)
    result = []
    for part, enc in decoded_parts:
        if isinstance(part, bytes):
            result.append(part.decode(enc or "utf-8", errors="replace"))
        else:
            result.append(str(part))
    return " ".join(result).strip()


def extract_body(msg: email.message.Message) -> str:
    """Extract plain text body from email message, falling back to html stripped if needed."""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))
            if content_type == "text/plain" and "attachment" not in content_disposition:
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    return payload.decode(charset, errors="replace")
        # If no plain text found, fallback to first text part
        for part in msg.walk():
            if part.get_content_type() == "text/html":
                payload = part.get_payload(decode=True)
                if payload:
                    # Simple strip tags
                    import re
                    html = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
                    return re.sub(r"<[^>]+>", " ", html)
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            return payload.decode(charset, errors="replace")
    return ""


def received_date_ist(date_header: str) -> Optional[str]:
    """Date header -> YYYY-MM-DD in IST (relative dates in the email are relative to the reader's day)."""
    try:
        dt = date_parser.parse(date_header)
    except (ValueError, OverflowError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=IST)
    return dt.astimezone(IST).date().isoformat()


def parse_mbox_file(
    mbox_path: Path,
    sender_filter: Optional[str] = "@iitj.ac.in",
    limit: Optional[int] = None,
    anonymize: bool = True,
    names_file: Optional[Path] = None,
) -> list[dict[str, Any]]:
    """Parse messages from an mbox file into records {id, received_date, sender, subject, body}."""
    names = load_names(names_file)
    records: list[dict[str, Any]] = []
    skipped = {"sender": 0, "date": 0, "body": 0, "error": 0}

    for idx, msg in enumerate(mailbox.mbox(str(mbox_path))):
        try:
            sender = decode_str(msg.get("From", ""))
            if sender_filter and sender_filter not in sender.lower():
                skipped["sender"] += 1
                continue
            received_date = received_date_ist(msg.get("Date", ""))
            if received_date is None:
                skipped["date"] += 1
                continue
            body = extract_body(msg).strip()
            if not body:
                skipped["body"] += 1
                continue
            record = {
                "id": f"real_{idx:05d}",
                "received_date": received_date,
                "sender": sender,
                "subject": decode_str(msg.get("Subject", "")),
                "body": body,
            }
            records.append(anonymize_record(record, names) if anonymize else record)
            if limit and len(records) >= limit:
                break
        except Exception as e:  # malformed message: count it, keep going
            skipped["error"] += 1
            print(f"  message {idx}: {type(e).__name__}: {e}")

    print(f"Parsed {len(records)} emails; skipped {skipped}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest mbox email archive into anonymized JSONL.")
    parser.add_argument("--mbox", type=str, required=True, help="Path to .mbox file")
    parser.add_argument("--out", type=str, default="data/real/raw_anonymized.jsonl")
    parser.add_argument("--filter", type=str, default="@iitj.ac.in")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--names-file", type=str, default="data/real/names_to_mask.txt",
                        help="Private list of names to mask, one per line (gitignored)")
    args = parser.parse_args()

    records = parse_mbox_file(Path(args.mbox), sender_filter=args.filter, limit=args.limit,
                              names_file=Path(args.names_file))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Extracted {len(records)} anonymized emails to {out_path}")


if __name__ == "__main__":
    main()
