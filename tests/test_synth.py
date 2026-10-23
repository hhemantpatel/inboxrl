"""Tests for synthetic email dataset generation and difficulty levels."""

from datetime import datetime
from inboxrl.data.synth import generate_dataset
from inboxrl.schema import Event


def test_generate_dataset_counts_and_structure():
    dataset = generate_dataset(num_samples=50, seed=123)
    assert len(dataset) == 50

    has_empty = False
    has_non_empty = False

    for sample in dataset:
        assert "id" in sample
        assert "received_date" in sample
        assert "sender" in sample
        assert "subject" in sample
        assert "body" in sample
        assert "gold" in sample
        assert "level" in sample
        assert sample["level"] in {1, 2, 3, 4, 5}

        # Validate date format
        received_dt = datetime.strptime(sample["received_date"], "%Y-%m-%d")

        if len(sample["gold"]) == 0:
            has_empty = True
        else:
            has_non_empty = True
            for g in sample["gold"]:
                ev = Event.model_validate(g)
                event_dt = datetime.strptime(ev.date, "%Y-%m-%d")
                # Event date should be on or after received date
                assert event_dt >= received_dt

    assert has_empty, "Dataset should include no-event emails (~15%)"
    assert has_non_empty, "Dataset should include events"


def test_reproducibility():
    ds1 = generate_dataset(num_samples=10, seed=42)
    ds2 = generate_dataset(num_samples=10, seed=42)
    assert ds1 == ds2
