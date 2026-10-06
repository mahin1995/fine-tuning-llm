# Qwen3 fine-tuning harness + tool-calling agent

Fine-tune `Qwen/Qwen3-0.6B` on a Java / Spring Boot Q&A dataset, chat with it from
the terminal or a browser, and run it as a tool-calling agent. Everything runs in
Docker on a single NVIDIA GPU (tested target: RTX 3060 12GB). See
[PROCESS.md](PROCESS.md) for the full log and the reasoning behind each decision.

## Project layout

```
data/                     datasets (Qwen chat format, one JSON object per line)
  train.jsonl             training data
  eval.jsonl              held-out questions with reference answers (never trained on)

qwen_ft/                  fine-tuning harness            -> python -m qwen_ft <command>
  config.py               shared constants (base model, paths, CHAT_TEMPLATE_KWARGS)
  data/                   schema validation, JSONL loading, prompt-completion + split   (no torch)
  modeling/               params (GenerationParams), loading (full / LoRA), chat_model (ChatModel)
  training/               options (TrainOptions dataclass), trainer (run_training)
  evaluation/             evaluator (model loader injected), report (markdown)
  serving/                app (FastAPI, model injected) + static/index.html
  cli/                    one module per command; the only place that wires packages together

agent/                    tool-calling agent loop         -> python -m agent
  backend.py              ChatBackend protocol: complete(messages, tools) -> str
  parser.py               <tool_call>{json}</tool_call> parsing
  tools.py                Tool, ToolRegistry (JSON schema from type hints, argument validation)
  loop.py                 Agent.run(question) -> AgentResult (answer, steps, transcript)
  builtin_tools.py        calculator (safe AST), current_time, search_knowledge_base
  backends/qwen.py        adapter to qwen_ft's ChatModel: the only bridge between the two packages

tests/                    CPU tests with a tiny local Qwen3 model (no download), incl. architecture rules
```

### Dependency rules (enforced by `tests/test_architecture.py`)

| Package | May import (inside this repo) |
|---|---|
| `qwen_ft.config` | nothing |
| `qwen_ft.data` | config |
| `qwen_ft.modeling` | config |
| `qwen_ft.training` | config, data |
| `qwen_ft.evaluation` | config, modeling.params (model loader is injected) |
| `qwen_ft.serving` | data, modeling.params (model is injected) |
| `qwen_ft.cli` | anything (composition root) |
| `agent` core (loop, tools, parser, builtin_tools) | nothing from `qwen_ft` |
| `agent/backends/qwen.py`, `agent/__main__.py` | `qwen_ft` (the only bridge) |

`qwen_ft` never imports `agent`. Lightweight modules (`data`, `config`, `modeling.params`, `evaluation`, `serving.app`,
`training.options`, the CLI dispatcher) must not import torch/transformers; tests check this too.

## Quick start

```bash
./run.sh build                                   # build image (pins library versions)
./run.sh python -m qwen_ft download              # download Qwen3-0.6B + sanity checks
./run.sh python -m qwen_ft validate              # check data/train.jsonl, data/eval.jsonl
./run.sh python -m pytest                        # test suite (~30s, CPU is enough)

./run.sh python -m qwen_ft train                 # full fine-tuning
./run.sh python -m qwen_ft train --lora --merge  # or: LoRA adapter (+ merged copy)
nohup ./run.sh python -m qwen_ft train > train.log 2>&1 &   # long runs in the background

./run.sh python -m qwen_ft evaluate --model outputs/qwen3-ft
./run.sh python -m qwen_ft chat --model outputs/qwen3-ft
PORT=8000 ./run.sh python -m qwen_ft serve --model outputs/qwen3-ft   # http://localhost:8000

./run.sh python -m agent --model outputs/qwen3-ft "What is 17 * 23?"  # one question
./run.sh python -m agent --model Qwen/Qwen3-0.6B                      # interactive
```

Smoke test on the 10-example dataset (the defaults give only 3 optimizer steps):

```bash
./run.sh python -m qwen_ft train --epochs 10 --batch-size 2 --grad-accum 1
```

## The agent loop

```
question -> model (with tool schemas) -> <tool_call>? --no--> final answer
                 ^                            | yes
                 |                            v
                 +---- "tool" message <--- validate args, run tool (errors go back to the model)
```

Safety rails: `--max-steps` limit, JSON-schema argument validation, tool exceptions
and malformed calls returned to the model as errors, identical repeated calls not
re-executed, tool output truncated. Built-in tools have no side effects.

Adding a tool:

```python
from agent import ToolRegistry, make_tool

def count_orders(status: str) -> int:
    """Count orders with the given status (NEW, PAID, SHIPPED)."""
    ...

registry = ToolRegistry([make_tool(count_orders, params={"status": "order status"})])
```

Using another model: implement `ChatBackend.complete(messages, tools) -> str` and pass
it to `Agent`. The loop doesn't know or care which model is behind it.

A 0.6B model is weak at tool calling out of the box; fine-tuning on tool-call
examples is the way to make it reliable (see PROCESS.md, next steps).

## Key design decisions

- **Prompt-completion training.** Only the final assistant answer contributes to
  the loss. Qwen3's template has no `{% generation %}` markers, so TRL's
  `assistant_only_loss` can't be used; splitting into prompt/completion gives the
  same effect.
- **`enable_thinking=False` everywhere** (`qwen_ft.config.CHAT_TEMPLATE_KWARGS`).
  Training, inference and the agent use the same setting so the formats match.
- **Full FT loads fp32 weights + bf16 autocast.** Pure-bf16 weights would round
  small updates away. 8-bit AdamW keeps it inside 12GB VRAM.
- **Dependency injection at the edges.** Training takes a `TrainOptions` dataclass
  (no argparse), evaluation takes a model loader, the server takes a model factory,
  the agent takes a `ChatBackend`. Tests swap in fakes; the CLI wires real objects.

## API

```bash
curl -s localhost:8000/chat -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "What is the N+1 problem?"}],
  "max_new_tokens": 256, "temperature": 0.7
}'
# {"reply": "..."}
```
