# Qwen Fine-Tuning Pipeline — Process Log

Full fine-tuning of a small Qwen model (`Qwen/Qwen3-0.6B`) on a Java/Spring Boot
Q&A dataset, served via a local FastAPI chat endpoint. Ollama is **not** used —
inference runs directly through `transformers`.

## System

| Item | Value |
|---|---|
| OS | Ubuntu Linux |
| GPU | NVIDIA GeForce RTX 3060, 12GB VRAM |
| Driver / CUDA | 595.91.07 / CUDA 13.2 |
| Python (host venv) | 3.14.4 — **unused**, superseded by Docker (see below) |
| Runtime | Docker container (`pytorch/pytorch:2.14.0-cuda13.2-cudnn9-runtime` base) |

## Why Docker instead of the host venv

The host venv (`qwen-finetune`, Python 3.14) was created first, but installing
`torch` via pip over a slow connection stalled repeatedly. We switched to
pulling a prebuilt `pytorch/pytorch` CUDA 13.2 image from Docker Hub instead —
same end result (torch + CUDA 13.2 working with the RTX 3060), pulled as
prebuilt layers rather than assembled via pip. The host venv is currently
unused; all work below runs inside the container.

## Step 1 — Environment Setup

**Docker + NVIDIA Container Toolkit** (one-time host setup, done via `sudo`):

```bash
sudo usermod -aG docker $USER          # add user to docker group
# NVIDIA Container Toolkit repo + install
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**Verified:**
```
$ docker info | grep -i runtime
Runtimes: io.containerd.runc.v2 nvidia runc

$ docker run --rm --gpus all pytorch/pytorch:2.14.0-cuda13.2-cudnn9-runtime \
  python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
2.14.0+cu132 13.2 True NVIDIA GeForce RTX 3060
```

GPU pass-through into the container confirmed working.

**Project image** — [Dockerfile](Dockerfile) adds the ML libraries on top of the
base image:

The library versions are pinned in [requirements.txt](requirements.txt). Pip
installs them with a constraint that locks the base image's CUDA torch, and the
build fails if a CPU-only torch ends up installed. Before pinning, a rebuild would
have pulled whatever was newest (transformers 5.18 had already been released),
and a dependency could have replaced torch with a different build.

Note: the base image's Python is Debian/Ubuntu's system Python (PEP 668,
"externally-managed-environment"), so plain `pip install` is blocked by
default. `--break-system-packages` is safe here because the container is
disposable and nothing else depends on its system Python.

[run.sh](run.sh) wraps `docker build` / `docker run`:
- Mounts the project directory at `/workspace` and the host's
  `~/.cache/huggingface` at `/hf` (so downloaded models survive container
  removal and are shared with the host).
- Runs as the host user (`--user $(id -u):$(id -g)`) so files written back to
  the project aren't root-owned.
- Falls back to `sudo docker` automatically if the `docker` group isn't active
  in the current shell yet.
- Uses `-it` only when a terminal is attached, so `nohup ./run.sh python -m qwen_ft train &` works.
- Publishes a port only when `PORT=...` is set, bound to `127.0.0.1`, so training
  and a debug shell can run side by side.
- Sets `USER` for libraries that look up the user name (the mapped uid has no passwd entry).

```bash
./run.sh build                      # build the qwen-ft image
./run.sh python ...                 # run a command inside it
./run.sh bash                       # interactive shell
PORT=8000 ./run.sh python -m qwen_ft serve  # with a published port
```

**Verified** — all libraries import correctly, and torch is still the CUDA
13.2 build (i.e. `trl`/`accelerate` didn't silently pull in a different
torch):
```
torch 2.14.0+cu132 True
transformers 5.17.0
trl 1.13.0
peft 0.21.0
bitsandbytes 0.50.2
```

## Step 2 — Dataset

[data/train.jsonl](data/train.jsonl) (originally `data.jsonl`) — 10 example Q&A pairs in Qwen chat format:
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Domain: Java/Spring Boot interview questions (stereotype annotations,
`@Transactional` propagation, the N+1 problem, bean scopes, constructor vs.
field injection, global exception handling, auto-configuration, JPA fetch
types, self-invocation proxy pitfall, rollback-on-checked-exceptions).

**Validated:** all 10 lines are valid JSON with the expected `user`/`assistant`
role structure. `python -m qwen_ft validate` now checks this automatically, along with
duplicates and train/eval leakage. With `--tokenizer` it also reports token lengths.

[data/eval.jsonl](data/eval.jsonl) holds 8 held-out Q&A pairs on topics not in the training
set (open-in-view, optimistic locking, `getReferenceById`, NESTED propagation, …),
used by `python -m qwen_ft evaluate`.

**Note on dataset size:** 10 examples is only enough to smoke-test the
pipeline. Training 3 epochs on this will make the model memorize these exact
Q&As rather than generalize. For real results:
- **~500–1,000 examples** — minimum to shift style/format
- **~2,000–5,000 examples** — noticeable domain improvement
- Full fine-tuning (vs. LoRA) on a very small dataset also risks catastrophic
  forgetting of the model's general abilities.

## Step 3 — Base Model Download

Checked the HF Hub API for both candidate models before downloading:

| Model | Gated | License | Params | Size on disk |
|---|---|---|---|---|
| `Qwen/Qwen3-0.6B` | No | apache-2.0 | 751.6M | ~1.5 GB (bf16 safetensors) |
| `Qwen/Qwen2.5-0.5B-Instruct` | No | apache-2.0 | 494.0M | ~1.0 GB (bf16 safetensors) |

`Qwen/Qwen3-0.6B` **is** available on Hugging Face Hub (not gated, not
Ollama-only), so it's the model used — no need to fall back to
Qwen2.5-0.5B-Instruct.

[qwen_ft/cli/download_model.py](qwen_ft/cli/download_model.py) downloads model + tokenizer via
`from_pretrained()`, loads weights in bf16, and prints:
- parameter count and bf16 memory footprint
- HF cache location and size on disk
- the Qwen chat template rendering (sanity check before training)

```bash
./run.sh python -m qwen_ft download
```

It also asserts that pad != eos and that the inference prompt
(`add_generation_prompt=True`, `enable_thinking=False`) is an exact prefix of the
training format. That check runs against the real Qwen3 template.

*(Status: script written and tested against a local tiny model; not yet run against the real Qwen3-0.6B download.)*

## Step 4 — Training Script *(written, tested on CPU)*

[qwen_ft/training/trainer.py](qwen_ft/training/trainer.py) uses TRL `SFTTrainer` and was checked against the installed
`transformers 5.17` / `trl 1.13` source:
- Data is converted to **prompt-completion** format, so loss is only on the final
  assistant answer. `assistant_only_loss` needs `{% generation %}` markers that
  Qwen3's template doesn't have.
- `chat_template_kwargs={"enable_thinking": False}` is passed per example, so training
  uses the same format as inference.
- **Full FT:** weights loaded in **fp32** with `bf16=True` autocast. Pure-bf16 weights
  would round small lr=2e-5 updates away. `adamw_8bit` plus gradient checkpointing
  keep this within 12GB.
- **`--lora`:** frozen bf16 base, r=16 adapters on all attention + MLP projections,
  lr 2e-4. `--merge` also writes a merged full model.
- Prints the number of optimizer steps and warns below 10. The original plan
  (batch 4 × accum 4 on 10 examples, 3 epochs) gives only **3 steps**.
- Catches CUDA OOM and prints concrete flags to retry with.
- Writes `run_info.json` (base model, args, metrics, library versions) next to the model.
- transformers 5 removed `warmup_ratio`; the script uses `warmup_steps=<float ratio>`.

## Step 5 — Training Run *(not started — needs the GPU host)*

```bash
./run.sh python -m qwen_ft train --epochs 10 --batch-size 2 --grad-accum 1   # smoke test, 10 examples
nohup ./run.sh python -m qwen_ft train > train.log 2>&1 &                    # real run
```

## Step 6 — Evaluation *(written, tested on CPU)*

[qwen_ft/evaluation/](qwen_ft/evaluation/) loads the base and fine-tuned models one at a time and,
for each `eval.jsonl` example, records:
- the **answer loss**: mean NLL of the reference answer, lower is better
- the greedy answer from each model

It writes `outputs/qwen3-ft/eval_report.md`.

## Step 7 — Chat CLI *(written, tested on CPU)*

- [qwen_ft/modeling/](qwen_ft/modeling/) holds the shared loading code (full model or LoRA
  adapter, merged on load) and generation (Qwen3 non-thinking sampling defaults,
  streaming, early stop).
- [qwen_ft/cli/chat.py](qwen_ft/cli/chat.py) is an interactive chat with streaming output, `/reset`,
  `/exit`, `--system` and bounded history.

## Step 8 — Chat API + Web UI *(written, tested on CPU)*

[qwen_ft/serving/app.py](qwen_ft/serving/app.py) is a FastAPI app with `GET /`
([qwen_ft/serving/static/index.html](qwen_ft/serving/static/index.html)), `GET /health` and `POST /chat`. Requests
are validated with the same rules as the training data, plus limits on message
count, content length, `max_new_tokens` and temperature. Generation runs in a
worker thread behind a lock, so the single GPU model is never called concurrently.

## Step 9 — Package restructure *(done)*

The flat scripts were split into two top-level packages with enforced boundaries
(see README "Project layout" for the full tree):
- `qwen_ft/` holds config, data, modeling, training, evaluation, serving and cli.
  Commands run as `python -m qwen_ft <command>`.
- Data moved to `data/train.jsonl` and `data/eval.jsonl`.
- Training logic takes a `TrainOptions` dataclass instead of argparse args, so it can
  be called from code. The evaluator receives a model loader and the server receives
  a model factory, so neither depends on how models are loaded.
- `tests/test_architecture.py` fails if a light module imports torch or a package
  imports a layer it shouldn't. Planting violations on purpose confirmed that it
  catches them.

## Step 10 — Agent loop *(written, tested on CPU)*

`agent/` is a model-agnostic tool-calling loop:
- `Agent.run()` asks the model, parses `<tool_call>{json}</tool_call>`, runs the tools,
  feeds the results back as `tool` messages, and repeats until the model answers
  without a tool call or `max_steps` is reached.
- `ToolRegistry` builds JSON schemas from type hints and validates the model's
  arguments before running anything.
- Errors (unknown tool, bad arguments, tool exception, malformed JSON) go back to the
  model as text instead of crashing the loop. Identical repeated calls are not
  re-executed, and tool output is truncated.
- Built-in tools have no side effects: a safe AST calculator (no `eval`, with limits on
  exponents and result size), `current_time`, and a keyword search over the training Q&A.
- `agent/backends/qwen.py` adapts `ChatModel`, whose `generate(..., tools=)` passes the
  schemas to the chat template, still with `enable_thinking=False`.
- CLI: `python -m agent [--model ...] ["question"]`; leave out the question for interactive mode.

**Caveat:** the real Qwen3-0.6B tool-calling behaviour has not been checked here (no
network access to Hugging Face). The test template mirrors Qwen3's tool format, and
`QwenBackend` warns if the real tokenizer treats `<tool_call>` as a special token,
since decoding would then strip it.

**Next step:** to make a 0.6B model reliable at tool calling, fine-tune it on tool-call
transcripts. The data schema would need to accept `tool` messages and assistant
`tool_calls`, and the training rows would carry a `tools` column, which TRL already reads.

## Step 11 — Hybrid CrewAI agent: Dataset Ops Assistant *(written, tested offline)*

`ops_crew/` handles natural-language requests about the training data: ask, dataset_stats,
add_example and remove_example. **The LLM proposes, deterministic code decides and executes.**

- **Deterministic layer** (`domain/`, plain Python, no CrewAI):
  - Pydantic validation of every task output, with 2 retries and then escalation
  - role- and business-rule policy that reuses `qwen_ft.data` (duplicates, eval leakage,
    and ids that must exist)
  - idempotency keys, plus an approval gateway for destructive or suspicious side effects
  - a JSON audit log with correlation ids
  - atomic dataset writes
- **Flow** (`flow.py`): a CrewAI 1.15 Flow with `@start`, `@router` and `@listen`. It only
  orders the steps; every decision comes from `domain/`.
- **Crew**: Intent Classifier → Dataset Researcher → Proposal Reviewer, run sequentially,
  temperature 0, `max_iter` / `max_execution_time` limits, `output_pydantic` on every
  task, and read-only traced tools.
- **Multi-LLM**: profiles in `llms.yaml` (local OpenAI-compatible server, OpenAI,
  Anthropic, in-process qwen_ft), selected per agent with env overrides, with a provider
  fallback chain. Keys come only from env vars, and missing keys fail at startup.
- **Verified CrewAI 1.15 behaviour** (by running it, not from docs):
  - Flows run offline.
  - A scripted `BaseLLM` can drive real agents and tools.
  - Unknown tools are refused.
  - Unparseable output raises `ConverterError`.
  - A route label that equals a method name is rejected.
- **Evals:** 13 golden cases (`python -m ops_crew.evals`) with a 0.8 pass threshold. An
  oracle crew passes all of them in the tests, which shows the expectations are consistent.
  Not yet run against a real LLM, since none is reachable from the dev container.
- **Bugs found by tests while building:**
  1. A resubmitted add was rejected as a "duplicate" before the idempotent replay could run.
     Fixed with an `already_executed` pass-through for role-checked replays.
  2. A missing API key surfaced as 3 retried crew failures. Fixed by checking every LLM at startup.

## Tests

`python -m pytest` runs 210 tests offline. The ops_crew tests skip themselves
when `requirements-crew.txt` isn't installed. They cover:
- **qwen_ft:** data, training (full and LoRA), template consistency, evaluation, serving
- **agent:** the loop, the parser, tools and calculator safety, Qwen integration
- **ops_crew:** the domain layer, every flow route with a mocked crew, real CrewAI
  agents on scripted LLMs, evals with an oracle crew
- **architecture:** package boundaries

## Open items / decisions made along the way

- Full fine-tuning remains the default, now with fp32 master weights, bf16 autocast
  and 8-bit AdamW. `--lora` is available and is the safer choice while the
  dataset is small.
- No GGUF/Ollama step — the fine-tuned model is served directly via
  `transformers` + FastAPI.
- All project files and the HF model cache live under this project directory
  / the host's `~/.cache/huggingface` (mounted into the container), not
  inside the disposable container filesystem.
