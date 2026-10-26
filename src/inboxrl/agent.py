"""Loading the trained agent (base model + GRPO LoRA adapter) and batched generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _dtype_for(device: str) -> torch.dtype:
    if device.startswith("cuda"):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load_model(
    base_model_name: str = DEFAULT_BASE_MODEL,
    adapter_path: Optional[str] = None,
    device: Optional[str] = None,
    merge_adapter: bool = True,
) -> Tuple[Any, Any]:
    """Load a base model, optionally fused with a trained LoRA adapter.

    Architecture & Inference Optimization:
        - LoRA Fusion: When `merge_adapter=True`, adapter weights Delta W = B * A are
          linearly fused into the base model weights W = W_0 + (alpha/r) * B * A via
          PEFT's `merge_and_unload()`. This eliminates adapter routing overhead during inference.
        - Left Padding: `tokenizer.padding_side = 'left'` is critical for decoder-only architectures
          (like Qwen) during batched autoregressive generation, ensuring the last token of each
          prompt sits at the frontier of attention without trailing pad token interference.
        - Strict Fallback Prevention: If `adapter_path` is specified, it MUST exist and contain
          `adapter_config.json`. We never silently fall back to the base model, preventing
          unintentional base-model evaluation masquerading as trained-agent evaluation.
    """
    device = device or default_device()
    if adapter_path is not None:
        adapter_dir = Path(adapter_path)
        if not (adapter_dir / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"No trained adapter at {adapter_dir} (adapter_config.json missing). "
                "Train one with `python -m inboxrl.train_grpo ... --promote`."
            )
        cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
        base_model_name = cfg.get("base_model_name_or_path") or base_model_name

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # decoder-only batched generation

    model = AutoModelForCausalLM.from_pretrained(base_model_name, dtype=_dtype_for(device))
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
        if merge_adapter:
            model = model.merge_and_unload()
    model.to(device)
    model.eval()
    return model, tokenizer


def load_agent(adapter_path: str = "checkpoints/final", device: Optional[str] = None) -> Tuple[Any, Any]:
    """Load the trained InboxRL policy (raises if the adapter is missing)."""
    return load_model(adapter_path=adapter_path, device=device)


@torch.no_grad()
def generate_batch(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    max_new_tokens: int = 192,
    batch_size: int = 8,
) -> list[str]:
    """Greedy-decode completions for already-templated prompt strings."""
    outputs: list[str] = []
    device = next(model.parameters()).device
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.pad_token_id,
        )
        gen = out[:, enc["input_ids"].shape[1] :]
        outputs.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    return [o.strip() for o in outputs]
