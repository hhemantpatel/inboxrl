"""Prompt definitions and email formatting utilities for InboxRL."""

from __future__ import annotations

SYSTEM_PROMPT = """You are an accurate calendar event extractor for academic institute emails.
Your task is to analyze the email and extract all scheduled events, deadlines, exams, classes, or meetings into a JSON array.

Output Schema:
A JSON array of event objects, where each object has:
- "title": (string) Concise event description including course/topic (e.g., "CSL3090 Assignment 2 Submission")
- "type": (string) Exactly one of: "deadline", "exam", "class", "event", "meeting"
- "date": (string) Date in "YYYY-MM-DD" format. Use the email's "Received Date" to resolve relative dates (e.g., "tomorrow", "next Friday").
- "start_time": (string or null) 24-hour start time in "HH:MM" format (e.g., "17:00", "23:59") or null if unspecified.
- "end_time": (string or null) 24-hour end time in "HH:MM" format or null if unspecified.
- "location": (string or null) Venue, room number, or meeting link if specified, else null.

Rules:
1. Output ONLY the valid JSON array starting with '[' and ending with ']'. No markdown commentary or extra text.
2. If the email contains NO calendar events, announcements with dates, or deadlines, output exactly: []
3. Never invent or hallucinate dates, times, or venues not mentioned or implied by the email.
"""

def format_email_user_content(received_date: str, sender: str, subject: str, body: str) -> str:
    """Format user turn content with email metadata and body."""
    # Truncate body if excessively long (e.g., ~1500 chars / ~400 tokens)
    clean_body = body.strip()
    if len(clean_body) > 2000:
        clean_body = clean_body[:2000] + "... [truncated]"
    
    return (
        f"Email Details:\n"
        f"- Received Date: {received_date}\n"
        f"- From: {sender}\n"
        f"- Subject: {subject}\n\n"
        f"Email Body:\n"
        f"{clean_body}\n\n"
        f"Extract events as a JSON array:"
    )

def build_chat_messages(received_date: str, sender: str, subject: str, body: str) -> list[dict[str, str]]:
    """Build conversation message history for chat template."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_email_user_content(received_date, sender, subject, body)},
    ]
