"""GRPO Reinforcement Learning Training for Calendar Event Extraction."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any, Optional

import torch
import yaml
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import GRPOConfig, GRPOTrainer

from inboxrl.prompts import build_chat_messages
from inboxrl.rewards import (
    reward_func_empty,
    reward_func_format,
    reward_func_match,
    reward_func_total,
)


class MetricsCSVCallback(TrainerCallback):
    """Callback to log step-by-step training metrics to a CSV file."""

    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = [
            "step",
            "reward",
            "reward_std",
            "kl",
            "completion_length",
            "loss",
            "learning_rate",
        ]
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def on_log(self, args: Any, state: Any, control: Any, logs: Optional[dict[str, Any]] = None, **kwargs: Any) -> None:
        if not logs:
            return

        row = {
            "step": state.global_step,
            "reward": logs.get("reward", logs.get("rewards/total", 0.0)),
            "reward_std": logs.get("reward_std", 0.0),
            "kl": logs.get("kl", logs.get("policy/kl", 0.0)),
            "completion_length": logs.get("completion_length", 0.0),
            "loss": logs.get("loss", 0.0),
            "learning_rate": logs.get("learning_rate", 0.0),
        }

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


def prepare_grpo_dataset(jsonl_path: Path, tokenizer: Any, max_samples: Optional[int] = None) -> Dataset:
    """Load JSONL dataset and format prompt strings using tokenizer chat template."""
    ds = load_dataset("json", data_files=str(jsonl_path), split="train")
    if max_samples is not None:
        ds = ds.select(range(min(len(ds), max_samples)))

    def format_row(example: dict[str, Any]) -> dict[str, Any]:
        messages = build_chat_messages(
            received_date=example["received_date"],
            sender=example["sender"],
            subject=example["subject"],
            body=example["body"],
        )
        prompt_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return {
            "prompt": prompt_str,
            "gold": example.get("gold", []),
            "level": example.get("level", 1),
        }

    return ds.map(format_row, batched=False)


def train(config: dict[str, Any], smoke_test: bool = False) -> None:
    seed = config.get("seed", 42)
    torch.manual_seed(seed)

    model_name = config.get("model_name_or_path", "Qwen/Qwen2.5-0.5B-Instruct")
    output_dir = Path(config.get("output_dir", "checkpoints/run"))
    final_adapter_dir = Path(config.get("final_adapter_dir", "checkpoints/final"))
    log_csv_path = Path(config.get("log_csv_path", "results/logs/train_metrics.csv"))

    output_dir.mkdir(parents=True, exist_ok=True)
    final_adapter_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    has_cuda = torch.cuda.is_available()
    device = "cuda" if has_cuda else "cpu"
    print(f"Training device: {device} (CUDA available: {has_cuda})")

    model_kwargs: dict[str, Any] = {}
    if has_cuda and config.get("load_in_4bit", False):
        print("Enabling BitsAndBytes 4-bit NormalFloat (NF4) quantization for 4GB VRAM...")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif has_cuda and config.get("torch_dtype") == "bfloat16":
        model_kwargs["torch_dtype"] = torch.bfloat16

    print(f"Loading base model: {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if has_cuda else None,
        **model_kwargs,
    )

    if config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()

    # Configure LoRA
    lora_config = LoraConfig(
        r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=config.get("lora_dropout", 0.05),
        target_modules=config.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        task_type="CAUSAL_LM",
        bias="none",
    )

    # Prepare datasets
    train_data_path = Path(config.get("data_path", "data/synthetic/train.jsonl"))
    max_samples = 16 if smoke_test else None
    print(f"Preparing dataset from {train_data_path} (max_samples={max_samples})...")
    train_dataset = prepare_grpo_dataset(train_data_path, tokenizer, max_samples=max_samples)

    # Sizing for smoke test vs full run
    max_steps = 10 if smoke_test else config.get("max_steps", 1200)
    num_generations = 2 if smoke_test else config.get("num_generations", 4)
    per_device_batch_size = 1 if smoke_test else config.get("per_device_train_batch_size", 1)
    grad_accum = 2 if smoke_test else config.get("gradient_accumulation_steps", 4)

    grpo_args = GRPOConfig(
        output_dir=str(output_dir),
        learning_rate=float(config.get("learning_rate", 2.0e-5)),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        warmup_steps=config.get("warmup_steps", 50) if not smoke_test else 2,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        num_generations=num_generations,
        max_prompt_length=config.get("max_prompt_length", 512),
        max_completion_length=config.get("max_completion_length", 192),
        temperature=config.get("temperature", 0.9),
        beta=config.get("beta", 0.04),
        logging_steps=config.get("logging_steps", 10) if not smoke_test else 1,
        save_steps=config.get("save_steps", 200) if not smoke_test else 5,
        save_total_limit=2,
        seed=seed,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[
            reward_func_total,
            reward_func_format,
            reward_func_match,
            reward_func_empty,
        ],
        args=grpo_args,
        train_dataset=train_dataset,
        peft_config=lora_config,
        callbacks=[MetricsCSVCallback(log_csv_path)],
    )

    print("Starting GRPO Reinforcement Learning Training...")
    trainer.train()

    print(f"Saving final trained LoRA adapter to {final_adapter_dir}...")
    trainer.save_model(str(final_adapter_dir))
    tokenizer.save_pretrained(str(final_adapter_dir))
    print("Training finished successfully!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train InboxRL model with GRPO.")
    parser.add_argument("--config", type=str, default="configs/grpo_qwen05b_4gb.yaml")
    parser.add_argument("--smoke-test", action="store_true", help="Run rapid 10-step smoke test")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    if args.max_steps is not None:
        config["max_steps"] = args.max_steps

    train(config, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
