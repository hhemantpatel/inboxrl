"""Prompt definitions and email formatting utilities for InboxRL."""

from __future__ import annotations

SYSTEM_PROMPT = """Extract every upcoming calendar event (deadline, exam, class, event, meeting) from the email.
Output ONLY a JSON array, no other text. Each item:
{"title": str, "type": "deadline"|"exam"|"class"|"event"|"meeting", "date": "YYYY-MM-DD", "start_time": "HH:MM" or null, "end_time": "HH:MM" or null, "location": str or null}
Rules:
- No events -> output []. Never invent dates, times or venues.
- Skip past, cancelled or replaced dates; keep only the new date.
- Resolve relative dates from the Received Date: today = that date, tomorrow = +1 day, "this <weekday>" = the coming <weekday> within 7 days, "next <weekday>" = 7 days after that.
- 24-hour times; midnight/EOD/end of day = 23:59, noon = 12:00.
- Title: short, include the course code if given (e.g. "CSL3090 Assignment 2 Submission")."""

MAX_BODY_CHARS = 2000


def format_email_user_content(received_date: str, sender: str, subject: str, body: str) -> str:
    """Format user turn content with email metadata and body."""
    clean_body = body.strip()
    if len(clean_body) > MAX_BODY_CHARS:
        clean_body = clean_body[:MAX_BODY_CHARS] + "... [truncated]"

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



def build_prompt(tokenizer, received_date: str, sender: str, subject: str, body: str, max_prompt_tokens: int = 640) -> str:
    """Render the chat prompt, cutting the email body until the prompt fits max_prompt_tokens.

    TRL >= 1.0 no longer truncates prompts, so we do it here (body only, never the instructions).
    """
    def render(b: str) -> str:
        return tokenizer.apply_chat_template(
            build_chat_messages(received_date, sender, subject, b), tokenize=False, add_generation_prompt=True
        )

    prompt = render(body)
    n_tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    if n_tokens <= max_prompt_tokens:
        return prompt

    body_ids = tokenizer(body, add_special_tokens=False)["input_ids"]
    overflow = n_tokens - max_prompt_tokens
    keep = len(body_ids) - overflow - 8  # margin for the truncation marker
    while keep > 0:
        prompt = render(tokenizer.decode(body_ids[:keep]) + " ... [truncated]")
        if len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) <= max_prompt_tokens:
            return prompt
        keep -= 16  # re-tokenisation after decode can shift the count slightly
    raise ValueError("max_prompt_tokens is too small to fit the system prompt")
