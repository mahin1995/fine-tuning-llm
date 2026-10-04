import sys

import torch
from huggingface_hub import scan_cache_dir
from huggingface_hub.errors import CacheNotFound
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_utils import CHAT_TEMPLATE_KWARGS
from inference import DEFAULT_BASE_MODEL

MODEL_ID = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASE_MODEL

print(f"Downloading {MODEL_ID} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)

n_params = sum(p.numel() for p in model.parameters())
print(f"\nparams:          {n_params / 1e6:.1f}M")
print(f"weights in RAM:  {n_params * 2 / 1e9:.2f} GB (bf16)")
print(f"vocab size:      {len(tokenizer)}")
print(f"eos / pad token: {tokenizer.eos_token!r} / {tokenizer.pad_token!r}")
# Training needs a pad token distinct from eos, otherwise eos gets masked and the model never learns to stop.
assert tokenizer.pad_token is not None, "tokenizer has no pad token"
assert tokenizer.pad_token != tokenizer.eos_token, "pad token equals eos token"

try:
    cached = [r for r in scan_cache_dir().repos if r.repo_id == MODEL_ID]
except CacheNotFound:  # local path given and the HF cache was never created
    cached = []
for repo in cached:
    print(f"cache on disk:   {repo.size_on_disk / 1e9:.2f} GB  ({repo.repo_path})")
if not cached:
    print("cache on disk:   not in the HF cache (loaded from a local path?)")

msgs = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
train_text = tokenizer.apply_chat_template(msgs, tokenize=False, **CHAT_TEMPLATE_KWARGS)
infer_text = tokenizer.apply_chat_template(msgs[:1], tokenize=False, add_generation_prompt=True,
                                           **CHAT_TEMPLATE_KWARGS)
print("\nchat template, training format:")
print(train_text)
print("chat template, inference prompt:")
print(infer_text)
# The prompt the model sees at inference must be exactly how training examples start.
assert train_text.startswith(infer_text), "inference prompt is not a prefix of the training format"
print("OK: model + tokenizer ready, train/inference formats match")
