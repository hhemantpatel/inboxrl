"""Tests for the formal MDP environment."""

from inboxrl.env import EmailExtractionEnv, EmailInstance


def test_env_lifecycle():
    sample = EmailInstance(
        id="test_01",
        received_date="2026-10-15",
        sender="prof@iitj.ac.in",
        subject="AI Assignment 2 Due",
        body="Submit AI Assignment 2 by 2026-10-22 at 23:59.",
        gold_events=[
            {
                "title": "AI Assignment 2 Due",
                "type": "deadline",
                "date": "2026-10-22",
                "start_time": "23:59",
                "end_time": None,
                "location": None,
            }
        ],
        difficulty_level=1,
    )

    env = EmailExtractionEnv(
        dataset=[sample],
        tokenizer=None,
        max_completion_tokens=5,
        eos_token_id=999,
    )

    initial_state = env.reset(0)
    assert initial_state == []

    # Step 1: token 10
    res1 = env.step(10)
    assert not res1.done
    assert res1.next_state == [10]
    assert res1.reward == 0.0

    # Step 2: EOS token (999) triggers terminal evaluation
    res2 = env.step(999)
    assert res2.done
    # Without a tokenizer, completion is empty, which evaluates against non-empty gold
    assert "breakdown" in res2.info
