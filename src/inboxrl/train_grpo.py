"""GRPO reinforcement learning for calendar event extraction (pure RL, no SFT).

Usage:
    python -m inboxrl.train_grpo --config configs/grpo_qwen05b_4gb.yaml --run-name main_s0 --seed 0 --promote
    python -m inboxrl.train_grpo --config configs/grpo_qwen05b_4gb.yaml --smoke-test

Each run writes to runs/<run-name>/: adapter/, metrics.csv, config.yaml, tensorboard logs.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import shutil
from pathlib import Path
from typing import Any, Optional

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from inboxrl.prompts import build_prompt
from inboxrl.rewards import (
    RewardWeights,
    make_reward_funcs,
    reward_func_empty,
    reward_func_format,
    reward_func_len,
    reward_func_match,
    reward_func_total,
    validate_gold,
)

# Order matters: reward_weights in the config refer to this list.
REWARD_FUNCS = [reward_func_total, reward_func_format, reward_func_match, reward_func_empty, reward_func_len]
DEFAULT_REWARD_WEIGHTS = [1.0, 0.0, 0.0, 0.0, 0.0]  # only R drives the advantage; components are logged

FINAL_ADAPTER_DIR = Path("checkpoints/final")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    for r in records:
        validate_gold(r["gold"])  # crash early on any malformed label
    return records


def to_prompt_rows(
    records: list[dict[str, Any]], tokenizer: Any, max_prompt_tokens: int, prompt_version: int = 1
) -> list[dict[str, Any]]:
    return [
        {
            "prompt": build_prompt(
                tokenizer, r["received_date"], r["sender"], r["subject"], r["body"], max_prompt_tokens, prompt_version
            ),
            "gold": r["gold"],
            "level": r.get("level", 0),
        }
        for r in records
    ]


def _cycle(rows: list[dict[str, Any]], n: int, rng: random.Random) -> list[dict[str, Any]]:
    """Draw n rows: repeated shuffled passes over `rows`."""
    out: list[dict[str, Any]] = []
    while len(out) < n:
        block = list(rows)
        rng.shuffle(block)
        out.extend(block)
    return out[:n]


def build_train_rows(
    rows: list[dict[str, Any]],
    real_rows: list[dict[str, Any]],
    cfg: dict[str, Any],
    prompts_per_step: int,
    max_steps: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Lay out the training stream in the exact order it will be consumed (shuffle_dataset=False).

    Curriculum: the first `curriculum_stage1_steps` optimizer steps only see easy emails
    (levels in `curriculum_stage1_levels`), the rest see the full mixture (+ real emails).
    Without curriculum the stream is a shuffled mixture from the start.
    """
    rng = random.Random(seed)
    total = prompts_per_step * max_steps
    real_frac = cfg.get("real_mix_fraction", 0.0) if real_rows else 0.0
    empty_frac = cfg.get("no_event_fraction")

    def draw(pool: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
        """n rows from `pool`; with `no_event_fraction` set, that share is no-event emails.

        The base model never answers `[]`, so at the natural ~15% rate most groups of a no-event prompt contain no
        `[]` sample and carry no signal. Oversampling them gives the rare `[]` samples more chances to be rewarded.
        """
        if empty_frac is None:
            return _cycle(pool, n, rng)
        empty = [r for r in pool if not validate_gold(r["gold"])]
        full = [r for r in pool if validate_gold(r["gold"])]
        n_empty = int(round(n * empty_frac)) if empty else 0
        out = _cycle(empty, n_empty, rng) + _cycle(full, n - n_empty, rng)
        rng.shuffle(out)
        return out

    def mixture(n: int) -> list[dict[str, Any]]:
        n_real = int(round(n * real_frac))
        mixed = draw(rows, n - n_real) + (_cycle(real_rows, n_real, rng) if n_real else [])
        rng.shuffle(mixed)
        return mixed

    if not cfg.get("use_curriculum", True):
        return mixture(total)

    stage1_steps = min(cfg.get("curriculum_stage1_steps", 400), max_steps)
    easy_levels = set(cfg.get("curriculum_stage1_levels", [1, 2]))
    easy = [r for r in rows if r["level"] in easy_levels]
    if not easy:
        raise ValueError(f"No training rows with levels {sorted(easy_levels)} for curriculum stage 1")
    n1 = stage1_steps * prompts_per_step
    return draw(easy, n1) + mixture(total - n1)


def load_config(path: Path) -> dict[str, Any]:
    """Read a YAML config; `base: other.yaml` inherits (recursively) from another config, keys here win."""
    with open(path, encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f) or {}
    if "base" in cfg:
        parent = load_config(Path(cfg.pop("base")))
        cfg = {**parent, **cfg}
    return cfg


def pick_precision() -> dict[str, Any]:
    """bf16 on Ampere+ GPUs, fp16 on older GPUs (e.g. Kaggle T4), fp32 on CPU."""
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return {"bf16": True, "fp16": False, "dtype": torch.bfloat16}
        return {"bf16": False, "fp16": True, "dtype": torch.float16}
    return {"bf16": False, "fp16": False, "dtype": torch.float32, "use_cpu": True}


def load_policy(cfg: dict[str, Any], dtype: torch.dtype) -> Any:
    model_name = cfg["model_name_or_path"]
    kwargs: dict[str, Any] = {"dtype": dtype}
    if cfg.get("load_in_4bit", False):
        if not torch.cuda.is_available():
            raise RuntimeError("load_in_4bit needs a CUDA GPU")
        if importlib.util.find_spec("bitsandbytes") is None:
            raise RuntimeError("load_in_4bit needs bitsandbytes: run `uv sync --extra qlora`")
        from peft import prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map={"": 0}, **kwargs)
        return prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.get("gradient_checkpointing", True))
    return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)


def write_metrics_csv(log_history: list[dict[str, Any]], path: Path) -> None:
    """Dump trainer.state.log_history (one row per logging step, train and eval) to CSV."""
    rows = [h for h in log_history if "step" in h]
    keys: list[str] = ["step"]
    for h in rows:
        for k in h:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def train(cfg: dict[str, Any], run_name: str, smoke_test: bool = False, promote: bool = False, resume: bool = False) -> Path:
    seed = int(cfg.get("seed", 0))
    run_dir = Path(cfg.get("runs_dir", "runs")) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if smoke_test:
        cfg.update(max_steps=10, num_generations=2, per_device_train_batch_size=2, gradient_accumulation_steps=1,
                   curriculum_stage1_steps=5, logging_steps=1, save_steps=1000, eval_steps=5, eval_samples=4,
                   warmup_steps=1, max_completion_length=min(cfg.get("max_completion_length", 192), 96))

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({**cfg, "run_name": run_name, "smoke_test": smoke_test}, f, sort_keys=False)

    precision = pick_precision()
    print(f"[inboxrl] device={'cuda:' + torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'} "
          f"bf16={precision['bf16']} fp16={precision['fp16']}")

    model_name = cfg["model_name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_prompt_tokens = int(cfg.get("max_prompt_tokens", 640))
    train_records = load_jsonl(Path(cfg["data_path"]))
    if smoke_test:
        train_records = train_records[:16]
    prompt_version = int(cfg.get("prompt_version", 1))
    rows = to_prompt_rows(train_records, tokenizer, max_prompt_tokens, prompt_version)

    real_rows: list[dict[str, Any]] = []
    real_path = cfg.get("real_train_path")
    if real_path and Path(real_path).exists() and not smoke_test:
        real_rows = to_prompt_rows(load_jsonl(Path(real_path)), tokenizer, max_prompt_tokens, prompt_version)
        print(f"[inboxrl] mixing {len(real_rows)} real training emails at fraction {cfg.get('real_mix_fraction', 0.0)}")

    num_generations = int(cfg.get("num_generations", 4))
    per_device_bs = int(cfg.get("per_device_train_batch_size", 4))
    grad_accum = int(cfg.get("gradient_accumulation_steps", 4))
    max_steps = int(cfg.get("max_steps", 800))
    world = max(torch.cuda.device_count(), 1)
    completions_per_step = per_device_bs * grad_accum * world
    if completions_per_step % num_generations:
        raise ValueError("per_device_train_batch_size * gradient_accumulation_steps must be divisible by num_generations")
    prompts_per_step = completions_per_step // num_generations

    train_dataset = Dataset.from_list(build_train_rows(rows, real_rows, cfg, prompts_per_step, max_steps, seed))
    print(f"[inboxrl] {prompts_per_step} prompts x {num_generations} generations per step, {max_steps} steps, "
          f"curriculum={'on' if cfg.get('use_curriculum', True) else 'off'}")

    eval_dataset = None
    eval_samples = int(cfg.get("eval_samples", 32))
    if cfg.get("val_data_path") and eval_samples > 0:
        val_records = load_jsonl(Path(cfg["val_data_path"]))[:eval_samples]
        eval_dataset = Dataset.from_list(to_prompt_rows(val_records, tokenizer, max_prompt_tokens, prompt_version))

    # `reward_params` overrides RewardWeights fields (e.g. date_closeness); empty -> the original reward.
    reward_funcs = make_reward_funcs(RewardWeights(**cfg.get("reward_params", {})))
    reward_weights = list(cfg.get("reward_weights", DEFAULT_REWARD_WEIGHTS))
    if len(reward_weights) != len(REWARD_FUNCS):
        raise ValueError(f"reward_weights needs {len(REWARD_FUNCS)} entries (total, format, match, empty, len)")

    report_to = "tensorboard" if importlib.util.find_spec("tensorboard") else "none"
    grpo_args = GRPOConfig(
        output_dir=str(run_dir),
        run_name=run_name,
        seed=seed,
        data_seed=seed,
        bf16=precision["bf16"],
        fp16=precision["fp16"],
        use_cpu=precision.get("use_cpu", False),
        learning_rate=float(cfg.get("learning_rate", 2e-5)),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        warmup_steps=int(cfg.get("warmup_steps", 50)),
        max_steps=max_steps,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
        optim=cfg.get("optim", "adamw_torch"),
        num_generations=num_generations,
        max_completion_length=int(cfg.get("max_completion_length", 192)),
        temperature=float(cfg.get("temperature", 0.9)),
        beta=float(cfg.get("beta", 0.04)),
        loss_type=cfg.get("loss_type", "dapo"),
        epsilon=float(cfg.get("epsilon", 0.2)),
        reward_weights=reward_weights,
        shuffle_dataset=False,  # build_train_rows already fixed the order (curriculum)
        log_completions=bool(cfg.get("log_completions", True)),
        logging_steps=int(cfg.get("logging_steps", 10)),
        save_steps=int(cfg.get("save_steps", 200)),
        save_total_limit=2,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=int(cfg.get("eval_steps", 100)),
        per_device_eval_batch_size=num_generations,
        report_to=report_to,
    )

    lora_config = LoraConfig(
        r=int(cfg.get("lora_r", 16)),
        lora_alpha=int(cfg.get("lora_alpha", 32)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        target_modules=cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
        task_type="CAUSAL_LM",
        bias="none",
    )

    trainer = GRPOTrainer(
        model=load_policy(cfg, precision["dtype"]),
        reward_funcs=reward_funcs,
        args=grpo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    try:
        trainer.train(resume_from_checkpoint=True if resume else None)
    finally:
        # Keep the metrics even if training crashes or is interrupted
        write_metrics_csv(trainer.state.log_history, run_dir / "metrics.csv")
        print(f"[inboxrl] metrics -> {run_dir / 'metrics.csv'}")

    adapter_dir = run_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"[inboxrl] adapter -> {adapter_dir}")

    if promote:
        if smoke_test:
            raise ValueError("Refusing to promote a smoke-test adapter to checkpoints/final")
        if FINAL_ADAPTER_DIR.exists():
            shutil.rmtree(FINAL_ADAPTER_DIR)
        shutil.copytree(adapter_dir, FINAL_ADAPTER_DIR)
        (FINAL_ADAPTER_DIR / "TRAINED_BY.txt").write_text(f"run: {run_name}\nconfig: see runs/{run_name}/config.yaml\n")
        (FINAL_ADAPTER_DIR / "inboxrl.json").write_text(json.dumps({
            "prompt_version": int(cfg.get("prompt_version", 1)),
            "max_prompt_tokens": int(cfg.get("max_prompt_tokens", 640)),
            "max_completion_length": int(cfg.get("max_completion_length", 192)),
        }))
        print(f"[inboxrl] promoted to {FINAL_ADAPTER_DIR}")
    return adapter_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the InboxRL policy with GRPO.")
    parser.add_argument("--config", type=str, default="configs/grpo_qwen05b_4gb.yaml")
    parser.add_argument("--run-name", type=str, default=None, help="Output folder under runs/ (default: config name + seed)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true", help="10 tiny steps to check the pipeline end to end")
    parser.add_argument("--promote", action="store_true", help="Copy the final adapter to checkpoints/final/")
    parser.add_argument("--resume", action="store_true", help="Resume from the last checkpoint in runs/<run-name>/")
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = load_config(config_path)

    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps

    run_name: Optional[str] = args.run_name
    if run_name is None:
        run_name = "smoke" if args.smoke_test else f"{config_path.stem}_s{cfg.get('seed', 0)}"

    train(cfg, run_name, smoke_test=args.smoke_test, promote=args.promote, resume=args.resume)


if __name__ == "__main__":
    main()
