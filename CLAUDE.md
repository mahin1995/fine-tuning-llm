# CLAUDE.md

Fine-tuning harness for Qwen3-0.6B (TRL SFT, full or LoRA) with eval, CLI chat and a FastAPI server.
README.md has the file map and commands; PROCESS.md has the decision log.

## Commands
- Tests (CPU, no network, ~10s): `python -m pytest` (or `./run.sh python -m pytest` in Docker)
- Data check: `python validate_data.py`
- Everything GPU-related runs through `./run.sh ...` (Docker, project mounted at /workspace)

## Rules that keep the pipeline correct
- Training and inference must render prompts identically: always pass
  `data_utils.CHAT_TEMPLATE_KWARGS` to `apply_chat_template`. Don't add a second copy of that setting.
- Library versions are pinned in `requirements.txt` and the APIs differ from older docs
  (transformers 5: `dtype=` not `torch_dtype=`, `warmup_steps` takes a ratio float; trl 1.x
  `SFTConfig`). Check the installed source before using an argument.
- Never let pip replace torch in the image; the Dockerfile constraint + CUDA check enforce it.
- `eval.jsonl` must never overlap `data.jsonl` (`validate_data.py` fails on leakage).
- Don't commit `outputs/` or weights (`.gitignore`).
- Tests use a tiny locally built Qwen3 model (`tests/conftest.py`); keep them network-free.
