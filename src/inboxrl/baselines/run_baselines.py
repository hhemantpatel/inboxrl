"""Run heuristic baseline evaluation."""

from pathlib import Path
from inboxrl.baselines.rule_based import RuleBasedExtractor
from inboxrl.evaluate import run_evaluation


def main():
    extractor = RuleBasedExtractor()

    def rule_extractor_func(record: dict) -> str:
        events = extractor.extract_events(
            received_date_str=record["received_date"],
            sender=record["sender"],
            subject=record["subject"],
            body=record["body"],
        )
        import json
        return json.dumps([ev.model_dump() for ev in events])

    metrics = run_evaluation(
        data_path=Path("data/synthetic/test.jsonl"),
        extractor_func=rule_extractor_func,
        out_json=Path("results/rule_based_test.json"),
    )

    print(f"Rule-Based Baseline Results:")
    print(f"  Valid JSON %:   {metrics.valid_json_ratio * 100:.1f}%")
    print(f"  Event F1:       {metrics.event_f1 * 100:.1f}%")
    print(f"  Date Match %:   {metrics.date_match_ratio * 100:.1f}%")
    print(f"  Mean Reward:    {metrics.mean_reward:.3f}")


if __name__ == "__main__":
    main()
