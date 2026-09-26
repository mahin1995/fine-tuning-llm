import sys

import torch
from huggingface_hub import scan_cache_dir
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"

print(f"Downloading {MODEL_ID} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)

n_params = sum(p.numel() for p in model.parameters())
print(f"\nparams:          {n_params / 1e6:.1f}M")
print(f"weights in RAM:  {n_params * 2 / 1e9:.2f} GB (bf16)")
print(f"vocab size:      {len(tokenizer)}")
print(f"eos / pad token: {tokenizer.eos_token!r} / {tokenizer.pad_token!r}")

for repo in scan_cache_dir().repos:
    if repo.repo_id == MODEL_ID:
        print(f"cache on disk:   {repo.size_on_disk / 1e9:.2f} GB  ({repo.repo_path})")

msgs = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
print("\nchat template preview:")
print(tokenizer.apply_chat_template(msgs, tokenize=False))
print("OK: model + tokenizer loaded from cache")
