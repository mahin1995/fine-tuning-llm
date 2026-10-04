# Qwen3 fine-tuning harness

Fine-tune `Qwen/Qwen3-0.6B` on a Java / Spring Boot Q&A dataset and chat with it
from the terminal or a browser. Everything runs in Docker on a single NVIDIA GPU
(tested target: RTX 3060 12GB). See [PROCESS.md](PROCESS.md) for the full log and
the reasoning behind each decision.

## Pipeline

```
data.jsonl ──validate_data.py──► train.py ──► outputs/qwen3-ft ──► evaluate.py (base vs fine-tuned)
                                  (full or LoRA)            ├────► chat.py   (terminal)
                                                            └────► serve.py  (API + web UI)
```

| File | Purpose |
|---|---|
| `data.jsonl` | training data, Qwen chat format (`{"messages": [...]}`) |
| `eval.jsonl` | held-out questions with reference answers (never trained on) |
| `data_utils.py` | load / validate / split data, prompt-completion conversion |
| `inference.py` | shared model loading (full model or LoRA adapter) and generation |
| `validate_data.py` | structure, duplicates, train/eval leakage, token lengths |
| `download_model.py` | download base model, check pad/eos and train/inference template match |
| `train.py` | SFT with TRL: full fine-tuning (default) or `--lora` |
| `evaluate.py` | answer loss + side-by-side answers, base vs fine-tuned |
| `chat.py` | interactive terminal chat with streaming |
| `serve.py` + `static/` | FastAPI `/chat` endpoint and a browser chat page |
| `tests/` | CPU test suite using a tiny local Qwen3 model (no download needed) |

## Quick start

```bash
./run.sh build                                   # build image (pins library versions)
./run.sh python download_model.py                # download Qwen3-0.6B + sanity checks
./run.sh python validate_data.py                 # check data.jsonl / eval.jsonl
./run.sh python -m pytest                        # test suite (~10s, CPU is enough)

./run.sh python train.py                         # full fine-tuning
./run.sh python train.py --lora --merge          # or: LoRA adapter (+ merged copy)
nohup ./run.sh python train.py > train.log 2>&1 &   # long runs in the background

./run.sh python evaluate.py --model outputs/qwen3-ft
./run.sh python chat.py --model outputs/qwen3-ft
PORT=8000 ./run.sh python serve.py --model outputs/qwen3-ft   # open http://localhost:8000
```

Smoke test on the 10-example dataset (the defaults give only 3 optimizer steps):

```bash
./run.sh python train.py --epochs 10 --batch-size 2 --grad-accum 1
```

## Key design decisions

- **Prompt-completion training.** Only the final assistant answer contributes to
  the loss. Qwen3's template has no `{% generation %}` markers, so TRL's
  `assistant_only_loss` can't be used; splitting into prompt/completion gives the
  same effect.
- **`enable_thinking=False` everywhere** (`data_utils.CHAT_TEMPLATE_KWARGS`). Qwen3
  inserts an empty `<think></think>` block in that mode; training and inference use
  the same setting so the formats match. `download_model.py` and the tests check this.
- **Full FT loads fp32 weights + bf16 autocast.** Pure-bf16 weights would round
  small updates away. 8-bit AdamW keeps it inside 12GB VRAM.
- **LoRA option** for small datasets: less forgetting, ~3-4GB VRAM, tiny output.

## API

```bash
curl -s localhost:8000/chat -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "What is the N+1 problem?"}],
  "max_new_tokens": 256, "temperature": 0.7
}'
# {"reply": "..."}
```
