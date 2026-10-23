"""MDP Formulation and Environment for InboxRL.

This module provides a formal Gym-like MDP interface for email calendar extraction:
- State: (system_prompt, email_context, reference_date, generated_token_ids)
- Action: token_id in vocabulary V
- Transition: deterministic concatenation of token_id
- Reward: Sparse terminal reward computed via Hungarian bipartite matching
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from inboxrl.prompts import build_chat_messages
from inboxrl.rewards import RewardBreakdown, RewardWeights, compute_reward_single
from inboxrl.schema import Event


@dataclass
class EmailInstance:
    id: str
    received_date: str
    sender: str
    subject: str
    body: str
    gold_events: list[dict[str, Any]] = field(default_factory=list)
    difficulty_level: int = 1


@dataclass
class StepResult:
    next_state: list[int]
    reward: float
    done: bool
    info: dict[str, Any]


class EmailExtractionEnv:
    """A formal sequential MDP environment for LLM-based structured extraction."""

    def __init__(
        self,
        dataset: list[EmailInstance],
        tokenizer: Any = None,
        max_completion_tokens: int = 192,
        eos_token_id: Optional[int] = None,
        weights: Optional[RewardWeights] = None,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_completion_tokens = max_completion_tokens
        self.eos_token_id = eos_token_id
        self.weights = weights or RewardWeights()

        self.current_idx = 0
        self.current_email: Optional[EmailInstance] = None
        self.prompt_tokens: list[int] = []
        self.generated_tokens: list[int] = []
        self.step_count = 0

    def reset(self, email_idx: Optional[int] = None) -> list[int]:
        """Reset the environment to the start of an email episode."""
        if email_idx is not None:
            self.current_idx = email_idx % len(self.dataset)
        else:
            self.current_idx = self.current_idx % len(self.dataset)

        self.current_email = self.dataset[self.current_idx]
        self.generated_tokens = []
        self.step_count = 0

        if self.tokenizer is not None:
            messages = build_chat_messages(
                self.current_email.received_date,
                self.current_email.sender,
                self.current_email.subject,
                self.current_email.body,
            )
            # Use tokenizer chat template if available
            if hasattr(self.tokenizer, "apply_chat_template"):
                self.prompt_tokens = list(self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True, return_dict=False
                ))
            else:
                text = f"{messages[0]['content']}\n\n{messages[1]['content']}"
                self.prompt_tokens = self.tokenizer.encode(text)
        else:
            self.prompt_tokens = []

        return self.state()

    def state(self) -> list[int]:
        """s_t = prompt tokens (system + email + received date) followed by the tokens generated so far."""
        return list(self.prompt_tokens) + list(self.generated_tokens)

    def step(self, action_token_id: int) -> StepResult:
        """Apply action (append token), check terminal condition, and compute reward."""
        if self.current_email is None:
            raise RuntimeError("Environment must be reset before calling step().")

        self.generated_tokens.append(action_token_id)
        self.step_count += 1

        is_eos = self.eos_token_id is not None and action_token_id == self.eos_token_id
        is_max_length = self.step_count >= self.max_completion_tokens
        done = is_eos or is_max_length

        reward = 0.0
        info: dict[str, Any] = {"step": self.step_count}

        if done:
            # Decode generated tokens to string
            if self.tokenizer is not None:
                completion_text = self.tokenizer.decode(
                    self.generated_tokens, skip_special_tokens=True
                )
            else:
                completion_text = ""

            breakdown: RewardBreakdown = compute_reward_single(
                completion=completion_text,
                gold=self.current_email.gold_events,
                weights=self.weights,
            )
            reward = breakdown.total
            info["breakdown"] = breakdown
            info["completion"] = completion_text

        return StepResult(
            next_state=self.state(),
            reward=reward,
            done=done,
            info=info,
        )
