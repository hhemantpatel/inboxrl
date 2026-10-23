"""Model and LoRA adapter loading utilities for InboxRL."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_agent(
    adapter_path: str = "checkpoints/final",
    base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    device: str = "cpu",
    load_in_4bit: bool = False,
    merge_adapter: bool = False,
) -> Tuple[Any, Any]:
    """Load model and tokenizer with optional LoRA adapter."""
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict[str, Any] = {"torch_dtype": "auto", "device_map": device}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="bfloat16",
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(base_model_name, **kwargs)

    adapter_p = Path(adapter_path)
    if adapter_p.exists() and (adapter_p / "adapter_config.json").exists():
        print(f"Loading LoRA adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, str(adapter_p))
        if merge_adapter:
            model = model.merge_and_unload()
    else:
        print(f"No LoRA adapter found at {adapter_path}. Using base model {base_model_name}.")

    return model, tokenizer
