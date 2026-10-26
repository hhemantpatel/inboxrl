"""LLM Prompting Baselines (Zero-Shot, Few-Shot) for Calendar Extraction."""

from __future__ import annotations

import json
from typing import Any, Optional

from inboxrl.prompts import SYSTEM_PROMPTS, format_email_user_content

FEW_SHOT_EXAMPLES = [
    {
        "received_date": "2026-10-05",
        "sender": "instructor@iitj.ac.in",
        "subject": "[CSL3090] Assignment 1 Due Date",
        "body": "Dear Students, please submit your Assignment 1 by 12 October 2026 at 23:59 on the LMS.",
        "completion": json.dumps([
            {
                "title": "CSL3090 Assignment 1 Submission",
                "type": "deadline",
                "date": "2026-10-12",
                "start_time": "23:59",
                "end_time": None,
                "location": None,
            }
        ]),
    },
    {
        "received_date": "2026-09-14",
        "sender": "acad@iitj.ac.in",
        "subject": "Mid-Sem Schedule",
        "body": "The mid-sem exam for CSL3010 is scheduled on 20 September 2026 from 10:00 AM at LHC-101.",
        "completion": json.dumps([
            {
                "title": "CSL3010 Mid-Sem Exam",
                "type": "exam",
                "date": "2026-09-20",
                "start_time": "10:00",
                "end_time": None,
                "location": "LHC-101",
            }
        ]),
    },
    {
        "received_date": "2026-08-20",
        "sender": "sports@iitj.ac.in",
        "subject": "Gymkhana Sports Complex Renovation",
        "body": "Please be informed that the badminton court will remain closed for maintenance until further notice.",
        "completion": "[]",
    },
]


def build_zero_shot_messages(
    received_date: str, sender: str, subject: str, body: str, version: int = 1
) -> list[dict[str, str]]:
    """Build messages for zero-shot prompt."""
    return [
        {"role": "system", "content": SYSTEM_PROMPTS[version]},
        {"role": "user", "content": format_email_user_content(received_date, sender, subject, body)},
    ]


def build_few_shot_messages(
    received_date: str, sender: str, subject: str, body: str, version: int = 1
) -> list[dict[str, str]]:
    """Build messages for 3-shot in-context learning prompt."""
    messages = [{"role": "system", "content": SYSTEM_PROMPTS[version]}]
    for ex in FEW_SHOT_EXAMPLES:
        messages.append({
            "role": "user",
            "content": format_email_user_content(ex["received_date"], ex["sender"], ex["subject"], ex["body"]),
        })
        messages.append({
            "role": "assistant",
            "content": ex["completion"],
        })
    messages.append({
        "role": "user",
        "content": format_email_user_content(received_date, sender, subject, body),
    })
    return messages


class PromptingBaseline:
    """Wrapper to run prompting inference on a HuggingFace causal LM."""

    def __init__(self, model: Any, tokenizer: Any, mode: str = "zero-shot", device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.mode = mode
        self.device = device

    def generate(self, received_date: str, sender: str, subject: str, body: str, max_new_tokens: int = 192) -> str:
        if self.mode == "few-shot":
            messages = build_few_shot_messages(received_date, sender, subject, body)
        else:
            messages = build_zero_shot_messages(received_date, sender, subject, body)

        prompt_str = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt_str, return_tensors="pt").to(self.device)

        import torch
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Slice out prompt tokens
        generated_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
