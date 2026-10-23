"""Ingest Google Takeout mbox files and filter to institute emails."""

from __future__ import annotations

import argparse
import email
import json
import mailbox
from datetime import datetime
from email.header import decode_header
from pathlib import Path
from typing import Any, Optional

from dateutil import parser as date_parser

from inboxrl.data.anonymize import anonymize_record


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
                    html = payload.decode("utf-8", errors="replace")
                    return re.sub(r"<[^>]+>", " ", html)
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            return payload.decode(charset, errors="replace")
    return ""


def parse_mbox_file(
    mbox_path: Path,
    sender_filter: Optional[str] = "@iitj.ac.in",
    limit: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Parse messages from an mbox file into structured dictionary records."""
    mbox = mailbox.mbox(str(mbox_path))
    records = []
    count = 0

    for idx, msg in enumerate(mbox):
        try:
            sender = decode_str(msg.get("From", ""))
            if sender_filter and sender_filter not in sender.lower():
                continue

            subject = decode_str(msg.get("Subject", ""))
            date_raw = msg.get("Date", "")
            
            # Parse received date
            try:
                parsed_dt = date_parser.parse(date_raw)
                received_date = parsed_dt.date().isoformat()
            except Exception:
                received_date = datetime.now().date().isoformat()

            body = extract_body(msg).strip()
            if not body:
                continue

            record = {
                "id": f"real_mbox_{idx:05d}",
                "received_date": received_date,
                "sender": sender,
                "subject": subject,
                "body": body,
            }
            record = anonymize_record(record)
            records.append(record)
            count += 1
            if limit and count >= limit:
                break
        except Exception:
            continue

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest mbox email archive into anonymized JSONL.")
    parser.add_argument("--mbox", type=str, required=True, help="Path to .mbox file")
    parser.add_argument("--out", type=str, default="data/real/raw_anonymized.jsonl")
    parser.add_argument("--filter", type=str, default="@iitj.ac.in")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    records = parse_mbox_file(Path(args.mbox), sender_filter=args.filter, limit=args.limit)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Extracted {len(records)} anonymized emails to {out_path}")


if __name__ == "__main__":
    main()
