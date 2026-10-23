"""Run every baseline on the available test splits.

    python -m inboxrl.baselines.run_baselines            # all baselines (needs the models; GPU recommended)
    python -m inboxrl.baselines.run_baselines --no-llm   # only empty / random / rule
"""

from __future__ import annotations

import argparse
from pathlib import Path

from inboxrl.evaluate import (
    SPLITS,
    empty_extractor,
    load_records,
    model_extractor,
    random_extractor,
    rule_extractor,
    run_evaluation,
)

LLM_BASELINES = [
    # (result name, mode, model id)
    ("zeroshot_0.5b", "zero-shot", "Qwen/Qwen2.5-0.5B-Instruct"),
    ("fewshot_0.5b", "few-shot", "Qwen/Qwen2.5-0.5B-Instruct"),
    ("zeroshot_1.5b", "zero-shot", "Qwen/Qwen2.5-1.5B-Instruct"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all InboxRL baselines.")
    parser.add_argument("--splits", nargs="+", default=["synth_test", "real_test"])
    parser.add_argument("--no-llm", action="store_true", help="Skip the language-model baselines")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    splits = [s for s in args.splits if Path(SPLITS[s]).exists()]
    skipped = sorted(set(args.splits) - set(splits))
    if skipped:
        print(f"Skipping missing splits: {skipped}")
    data = {s: load_records(s) for s in splits}

    for split, records in data.items():
        run_evaluation(records, empty_extractor, "empty", split)
        run_evaluation(records, random_extractor, "random", split)
        run_evaluation(records, rule_extractor, "rule", split)

    if args.no_llm:
        return
    for name, mode, model_id in LLM_BASELINES:
        extractor = model_extractor(mode, model_id, args.batch_size, max_new_tokens=192, max_prompt_tokens=640)
        for split, records in data.items():
            run_evaluation(records, extractor, name, split)


if __name__ == "__main__":
    main()
