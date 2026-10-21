"""Load a model for inference: a Hub id, a full checkpoint dir, or a LoRA adapter dir."""
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def pick_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def inference_dtype(device):
    # bf16 halves memory on GPU; on CPU bf16 matmuls are slow, so stay in fp32.
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def is_adapter_dir(model_path):
    return (Path(model_path) / "adapter_config.json").is_file()


def load_model_and_tokenizer(model_path, device=None):
    device = device or pick_device()
    dtype = inference_dtype(device)
    if is_adapter_dir(model_path):
        from peft import AutoPeftModelForCausalLM

        model = AutoPeftModelForCausalLM.from_pretrained(model_path, dtype=dtype)
        model = model.merge_and_unload()  # plain model: faster generation, no peft overhead
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype)
    # Training saves the tokenizer next to the weights/adapter, so this works for all three cases.
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.to(device).eval()
    return model, tokenizer
