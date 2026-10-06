# CLAUDE.md

Fine-tuning harness for Qwen3-0.6B (`qwen_ft/`: TRL SFT, full or LoRA, eval, CLI chat, FastAPI)
plus a model-agnostic tool-calling agent loop (`agent/`).
README.md has the layout, dependency rules and commands; PROCESS.md has the decision log.

## Commands
- Tests (CPU, no network, ~30s): `python -m pytest` (or `./run.sh python -m pytest` in Docker)
- Data check: `python -m qwen_ft validate`
- All commands: `python -m qwen_ft --help`, `python -m agent --help`
- Everything GPU-related runs through `./run.sh ...` (Docker, project mounted at /workspace)

## Architecture rules (enforced by tests/test_architecture.py; keep them green)
- `qwen_ft` layering: config <- data/modeling <- training/evaluation/serving <- cli.
  Only `qwen_ft/cli/` wires concrete implementations together; inject dependencies elsewhere.
- Lightweight modules (data, config, modeling.params, evaluation, serving.app, training.options,
  the CLI dispatcher) must not import torch/transformers at module level. Import heavy libs lazily.
- `agent` core (loop, tools, parser, builtin_tools) never imports `qwen_ft` or ML libraries.
  Only `agent/backends/*` and `agent/__main__.py` may import `qwen_ft`. `qwen_ft` never imports `agent`.

## Rules that keep the pipeline correct
- Training, inference and the agent must render prompts identically: always pass
  `qwen_ft.config.CHAT_TEMPLATE_KWARGS` to `apply_chat_template`. Don't add a second copy of that setting.
- Library versions are pinned in `requirements.txt` and the APIs differ from older docs
  (transformers 5: `dtype=` not `torch_dtype=`, `warmup_steps` takes a ratio float; trl 1.x
  `SFTConfig`). Check the installed source before using an argument.
- Never let pip replace torch in the image; the Dockerfile constraint + CUDA check enforce it.
- `data/eval.jsonl` must never overlap `data/train.jsonl` (`validate` fails on leakage).
- Agent tools treat model arguments as untrusted: declare typed parameters so `ToolRegistry`
  validates them, raise `ToolError` for model-facing errors, and keep side effects out of
  built-in tools.
- Don't commit `outputs/` or weights (`.gitignore`).
- Tests use a tiny locally built Qwen3 model (`tests/conftest.py`); keep them network-free.
