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

```dockerfile
FROM pytorch/pytorch:2.14.0-cuda13.2-cudnn9-runtime

# Throwaway container: overriding PEP 668 is safe here. `python -m pip` makes
# sure packages land in the same interpreter that already has torch.
RUN python -m pip install --no-cache-dir --break-system-packages \
    --default-timeout=100 --retries 10 \
    transformers trl datasets accelerate peft bitsandbytes \
    fastapi "uvicorn[standard]"

WORKDIR /workspace
```

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
- Exposes port 8000 for the future chat API.

```bash
./run.sh build      # build the qwen-ft image
./run.sh python ...  # run a command inside it
./run.sh bash        # interactive shell
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

[data.jsonl](data.jsonl) — 10 example Q&A pairs in Qwen chat format:
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Domain: Java/Spring Boot interview questions (stereotype annotations,
`@Transactional` propagation, the N+1 problem, bean scopes, constructor vs.
field injection, global exception handling, auto-configuration, JPA fetch
types, self-invocation proxy pitfall, rollback-on-checked-exceptions).

**Validated:** all 10 lines are valid JSON with the expected `user`/`assistant`
role structure.

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

[download_model.py](download_model.py) downloads model + tokenizer via
`from_pretrained()`, loads weights in bf16, and prints:
- parameter count and bf16 memory footprint
- HF cache location and size on disk
- the Qwen chat template rendering (sanity check before training)

```bash
./run.sh python download_model.py
```

*(Status: script written, not yet confirmed run — awaiting output.)*

## Step 4 — Training Script *(not yet started)*

Planned: `train.py` using `trl`'s `SFTTrainer`, with:
- `per_device_train_batch_size=4`, `gradient_accumulation_steps=4`
- `gradient_checkpointing=True`, `bf16=True`
- `learning_rate=2e-5`, `num_train_epochs=3`
- `adamw_8bit` optimizer (via `bitsandbytes`) to keep full fine-tuning inside
  the RTX 3060's 12GB VRAM
- OOM handling that suggests lowering batch size
- Will be written against the actually-installed `transformers 5.17` /
  `trl 1.13` APIs (both are recent major versions, so signatures are checked
  against installed source rather than assumed).

## Step 5 — Training Run *(not started)*

Run `train.py` inside the container, monitor `nvidia-smi` and the loss curve,
retry with a smaller batch size on OOM, confirm the saved checkpoint path.

## Step 6 — Evaluation *(not started)*

Load the fine-tuned checkpoint, compare outputs against the base model on a
few held-out prompts.

## Step 7 — Chat CLI *(not started, replaces GGUF conversion)*

Ollama was uninstalled, so the GGUF/`ollama create` steps from the original
plan are dropped. Instead:
- `inference.py` — shared model-loading/generation code
- `chat.py` — interactive terminal chat using the Qwen chat template
  (`enable_thinking=False` by default for Qwen3)

## Step 8 — Chat API + Web UI *(not started)*

`serve.py` — FastAPI app exposing a `/chat` endpoint, plus a simple HTML page
served at `/` for chatting from the browser. Port 8000 is already forwarded by
`run.sh`.

## Open items / decisions made along the way

- Full fine-tuning (not LoRA) was chosen, made feasible on 8–12GB VRAM via
  bf16 weights + 8-bit AdamW optimizer states.
- No GGUF/Ollama step — the fine-tuned model is served directly via
  `transformers` + FastAPI.
- All project files and the HF model cache live under this project directory
  / the host's `~/.cache/huggingface` (mounted into the container), not
  inside the disposable container filesystem.
