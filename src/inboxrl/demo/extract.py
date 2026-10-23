"""End-to-end extraction CLI: Email/MBox -> Calendar Events -> .ics File."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from inboxrl.demo.ics import save_ics_file
from inboxrl.prompts import build_chat_messages
from inboxrl.schema import Event, parse_completion


def extract_from_text(
    model: Any,
    tokenizer: Any,
    email_text: str,
    received_date: Optional[str] = None,
    sender: str = "instructor@iitj.ac.in",
    subject: str = "Course Announcement",
    device: str = "cpu",
) -> list[Event]:
    if received_date is None:
        received_date = datetime.now().strftime("%Y-%m-%d")

    messages = build_chat_messages(received_date, sender, subject, email_text)
    prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_str, return_tensors="pt").to(device)

    import torch
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=192,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
    output_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    events = parse_completion(output_text)
    return events if events is not None else []


def main() -> None:
    parser = argparse.ArgumentParser(description="InboxRL: Extract calendar events from emails into .ics")
    parser.add_argument("--text", type=str, help="Raw email text to process")
    parser.add_argument("--mbox", type=str, help="Path to mbox email archive")
    parser.add_argument("--received-date", type=str, default=None, help="Reference date (YYYY-MM-DD)")
    parser.add_argument("--model-path", type=str, default="checkpoints/final", help="Path to trained model or LoRA")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--out", type=str, default="events.ics", help="Output .ics file path")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    if not args.text and not args.mbox:
        print("Please provide either --text '<email body>' or --mbox <path_to_mbox>")
        sys.exit(1)

    print("Loading model and tokenizer...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype="auto",
        device_map=args.device,
    )

    if Path(args.model_path).exists():
        print(f"Loading trained LoRA adapter from {args.model_path}...")
        model = PeftModel.from_pretrained(model, args.model_path)

    all_events: list[Event] = []

    if args.text:
        events = extract_from_text(
            model=model,
            tokenizer=tokenizer,
            email_text=args.text,
            received_date=args.received_date,
            device=args.device,
        )
        all_events.extend(events)
        print("\n--- Extracted Events ---")
        for ev in events:
            print(json.dumps(ev.model_dump(), indent=2))

    elif args.mbox:
        from inboxrl.data.ingest_mbox import parse_mbox_file
        records = parse_mbox_file(Path(args.mbox))
        print(f"Processing {len(records)} emails from mbox...")
        for r in records:
            events = extract_from_text(
                model=model,
                tokenizer=tokenizer,
                email_text=r["body"],
                received_date=r["received_date"],
                sender=r["sender"],
                subject=r["subject"],
                device=args.device,
            )
            all_events.extend(events)

    if all_events:
        save_ics_file(all_events, Path(args.out))
        print(f"\nSaved {len(all_events)} events to {args.out}")
    else:
        print("\nNo events detected in provided email(s).")


if __name__ == "__main__":
    main()
