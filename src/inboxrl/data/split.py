"""Split labeled datasets into train, validation, and held-out test splits."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def split_data(
    input_file: Path,
    out_dir: Path,
    test_ratio: float = 0.7,
    seed: int = 42,
) -> None:
    with open(input_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    rng = random.Random(seed)
    rng.shuffle(records)

    n_test = int(len(records) * test_ratio)
    test_records = records[:n_test]
    train_records = records[n_test:]

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "real_test.jsonl", "w", encoding="utf-8") as f:
        for r in test_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(out_dir / "real_train.jsonl", "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Split {len(records)} records into:")
    print(f"  - Test:  {len(test_records)} -> {out_dir / 'real_test.jsonl'}")
    print(f"  - Train: {len(train_records)} -> {out_dir / 'real_train.jsonl'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split labeled emails into train and test sets.")
    parser.add_argument("--input", type=str, default="data/real/labeled.jsonl")
    parser.add_argument("--out-dir", type=str, default="data/real")
    parser.add_argument("--test-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split_data(Path(args.input), Path(args.out_dir), args.test_ratio, args.seed)


if __name__ == "__main__":
    main()
