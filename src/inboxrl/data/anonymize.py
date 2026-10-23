"""Anonymisation of institute emails before they are stored or committed.

Masks roll numbers, phone numbers, personal (non-institute) email addresses, student
institute addresses, sender display names, and any names listed in a private names
file (one name per line; keep that file out of git). Always review the output by hand.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Optional

ROLL_NUMBER_REGEX = re.compile(r"\b[BMDP]\d{2}[A-Z]{2,3}\d{3,4}\b", re.IGNORECASE)
PHONE_REGEX = re.compile(r"(?<!\d)(?:\+91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")
EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
INSTITUTE_DOMAIN = "iitj.ac.in"
# Role / office mailboxes that identify no individual and help the model (e.g. acad-office@)
ROLE_LOCALPART = re.compile(r"^(?:[a-z]+[-_])*(?:office|acad|cdc|gymkhana|secretary|library|admin|hostel|dean|registrar|"
                            r"head|hod|sports|mess|cc|noreply|no-reply|info|council|club|placement)(?:[-_.][a-z]+)*$",
                            re.IGNORECASE)


def load_names(path: Optional[Path]) -> list[str]:
    if path is None or not path.exists():
        return []
    names = [n.strip() for n in path.read_text(encoding="utf-8").splitlines() if n.strip() and not n.startswith("#")]
    return sorted(names, key=len, reverse=True)  # longest first so "Hemant Patel" wins over "Hemant"


def _mask_email(match: re.Match[str]) -> str:
    addr = match.group(0)
    local, _, domain = addr.partition("@")
    if domain.lower() != INSTITUTE_DOMAIN:
        return "[EXTERNAL_EMAIL]"
    if ROLE_LOCALPART.match(local):
        return addr
    return "[PERSON_EMAIL]"


def anonymize_text(text: str, names: Iterable[str] = ()) -> str:
    """Mask personal identifiers in free text."""
    if not text:
        return text
    text = EMAIL_REGEX.sub(_mask_email, text)
    text = ROLL_NUMBER_REGEX.sub("[ROLL_NUMBER]", text)
    text = PHONE_REGEX.sub("[PHONE_NUMBER]", text)
    for name in names:
        text = re.sub(rf"\b{re.escape(name)}\b", "[NAME]", text, flags=re.IGNORECASE)
    return text


def anonymize_sender(sender: str) -> str:
    """'Some Person <addr>' -> masked address only (display names are dropped)."""
    m = EMAIL_REGEX.search(sender)
    if not m:
        return "[SENDER]"
    return _mask_email(m)


def anonymize_record(record: dict[str, Any], names: Iterable[str] = ()) -> dict[str, Any]:
    names = list(names)
    out = dict(record)
    out["subject"] = anonymize_text(record.get("subject", ""), names)
    out["body"] = anonymize_text(record.get("body", ""), names)
    out["sender"] = anonymize_sender(record.get("sender", ""))
    return out
