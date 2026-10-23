"""Data anonymization utilities for institute emails to protect student privacy."""

from __future__ import annotations

import re
from typing import Any

# Regex patterns for identifying personal sensitive data
ROLL_NUMBER_REGEX = re.compile(r"\b[BbMmDdPp]\d{2}[A-Za-z]{2}\d{4}\b", re.IGNORECASE)
PHONE_REGEX = re.compile(r"(?:\+91[\s-]?)?[6-9]\d{9}\b")
PERSONAL_EMAIL_REGEX = re.compile(
    r"\b[A-Za-z0-9._%+-]+@(?!iitj\.ac\.in)(?:gmail|yahoo|outlook|hotmail|icloud)\.[A-Za-z]{2,}\b",
    re.IGNORECASE,
)
STUDENT_IITJ_EMAIL_REGEX = re.compile(
    r"\b([a-z]+(?:\.[a-z]+)?)\d{0,4}@iitj\.ac\.in\b",
    re.IGNORECASE,
)


def anonymize_text(text: str) -> str:
    """Mask student roll numbers, phone numbers, and personal emails from text."""
    if not text:
        return text

    # Mask roll numbers (e.g., B23CH1018 -> [ROLL_REDACTED])
    text = ROLL_NUMBER_REGEX.sub("[ROLL_NUMBER]", text)

    # Mask phone numbers
    text = PHONE_REGEX.sub("[PHONE_NUMBER]", text)

    # Mask personal external emails
    text = PERSONAL_EMAIL_REGEX.sub("[EXTERNAL_EMAIL]", text)

    return text


def anonymize_record(record: dict[str, Any]) -> dict[str, Any]:
    """Anonymize subject, body, and sender of an email record."""
    anonymized = dict(record)
    anonymized["subject"] = anonymize_text(record.get("subject", ""))
    anonymized["body"] = anonymize_text(record.get("body", ""))

    sender = record.get("sender", "")
    # If sender has a student roll or name pattern, mask it
    if ROLL_NUMBER_REGEX.search(sender):
        anonymized["sender"] = ROLL_NUMBER_REGEX.sub("[STUDENT_SENDER]", sender)

    return anonymized
