# Qwen3 fine-tuning harness + agents

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

ops_crew/                 hybrid CrewAI agent: Dataset Ops Assistant  -> python -m ops_crew "<request>"
  schemas.py              Pydantic contracts for every crew output
  settings.py             OPS_* env settings (thresholds, paths, retries, timeouts)
  config/                 agents.yaml, tasks.yaml, llms.yaml (multi-LLM profiles)
  domain/                 deterministic layer, plain Python: validation, policy, idempotency,
                          approval, audit, repository, executor            (no CrewAI)
  flow.py                 CrewAI Flow: intake -> crew -> validate -> authorize -> approve -> execute
  crew/                   3-agent sequential crew, read-only tools, LLM factory with fallbacks
  evals/                  14 golden cases + runner              -> python -m ops_crew.evals
  refinement.py           adapter to `refiner` (the only module that imports it)

refiner/                  generic self-correction + reflection on PydanticAI (domain-agnostic)
  correction.py           self_correct(): typed output, schema/check problems fed back as ModelRetry
  reflection.py           reflect(): critic -> reviser loop, best-so-far, stop rules
  models.py               OpenAI-compatible / Ollama / local function models, output modes, fallback

tests/                    CPU tests (tiny local Qwen3 model, scripted LLMs for CrewAI), incl. architecture rules
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
| `refiner` | only `pydantic` / `pydantic_ai` (no project package) |
| `ops_crew/refinement.py` | `refiner` (the only module that may) |

`qwen_ft` never imports `agent`. `ops_crew.domain` imports no CrewAI and only reuses
`qwen_ft.data`; only `ops_crew/crew/`, `flow.py`, `evals/` and `__main__` import CrewAI;
`qwen_ft`, `agent` and `ops_crew` never import each other sideways. Lightweight modules (`data`, `config`, `modeling.params`, `evaluation`, `serving.app`,
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

## Hybrid agent: Dataset Ops Assistant (`ops_crew`)

Natural-language requests about the training data, handled with a strict split:
**the LLM crew proposes, deterministic code validates, decides and executes.**

| Intent | Example | Side effect | Needs |
|---|---|---|---|
| `ask` | "What does REQUIRES_NEW do?" | none | viewer |
| `dataset_stats` | "How many examples are there?" | none (numbers come from code) | viewer |
| `add_example` | "Add Q: ... A: ..." | append to `data/train.jsonl` (idempotent) | editor |
| `remove_example` | "Remove the bean-scope example" | delete one row (atomic write) | admin + human approval |

```
intake ─► check_input ─┬─ input_invalid ──► reject_input
                       └─ input_ok ──► propose: crew + Pydantic validation (1 try + 2 retries)
                                         ├─ needs_escalation ──► escalate (human)
                                         ├─ low_confidence ────► clarify (question back to user)
                                         └─ proposal_ready ────► authorize_proposal (policy.py)
                                                                   ├─ denied ──────────► reject_proposal
                                                                   ├─ authorized ──────► execute_action ─► audit
                                                                   └─ approval_required ► request_approval
                                                                                           ├─ approved ─► execute_action
                                                                                           └─ declined ─► decline
```

**Crew** (sequential, `allow_delegation=False`, `temperature=0`, `max_iter` + `max_execution_time` per agent,
`output_pydantic` on every task):
Intent Classifier (no tools) -> Dataset Researcher (search, get, duplicate, eval-overlap, stats; all read-only)
-> Proposal Reviewer (get, duplicate). Sequential because the work is a fixed pipeline; a hierarchical
manager would add LLM calls and unpredictability for no gain.

**Multi-LLM** (`ops_crew/config/llms.yaml`): profiles `local` (any OpenAI-compatible server: Ollama, vLLM, ...),
`openai`, `anthropic`, `qwen_ft` (this repo's model, experimental). Per agent:
`OPS_LLM_PROFILE_<AGENT>` > `OPS_LLM_PROFILE` > `llm:` in agents.yaml. Profiles list `fallbacks` that are tried
when a provider fails (skipped if their key isn't set). Keys come only from env vars named in the profile.

```bash
./run.sh python -m ops_crew "How many examples are in the training data?"
./run.sh python -m ops_crew --role editor "Add Q: What is a Spring profile? A: A named set of beans ..."
./run.sh python -m ops_crew --role admin "Remove the example about bean scopes"     # asks for approval
OPS_LLM_PROFILE=openai OPENAI_API_KEY=... ./run.sh python -m ops_crew.evals --report outputs/ops/evals.json
OPS_LLM_PROFILE_REVIEWER=anthropic ANTHROPIC_API_KEY=... ./run.sh python -m ops_crew "..."   # mix providers
```

Audit log: `outputs/ops/audit.jsonl` (one JSON record per step, all with the run's correlation id).
Idempotency store: `outputs/ops/idempotency.jsonl`; pass `--request-id` to scope it explicitly.

| Edge case | Handling |
|---|---|
| Invalid JSON / schema violation | Pydantic validation, retry (max 2), then escalate |
| Hallucinated tool | CrewAI refuses unknown tools; config validation rejects unknown tool names |
| Hallucinated example id | Policy checks the id exists before anything runs |
| Agent loop / timeout | `max_iter`, `max_execution_time`, plus a per-attempt crew timeout in the flow |
| Duplicate submissions / retries | Idempotency key per action; replays return the stored result |
| Prompt injection | Delimited, neutralised user and tool text; read-only tools; role from caller; markers force approval for side effects |
| Low confidence | Clarifying question, or escalation if there is none |
| LLM provider failure | Provider fallback chain, then retry, then escalation; missing keys fail at startup |

## Self-correction and reflection (`refiner` + `ops_crew/refinement.py`)

Blind retries re-send the same prompt; at temperature 0 the model tends to repeat the same
mistake. `refiner` (PydanticAI) instead tells the model exactly what was wrong:

```
crew proposal ─► validate each task output ─► correctable problem? ─yes─► REPAIR (self_correct)
                                               │                           schema errors + check problems
                                               no                          sent back as feedback (max 2)
                                               ▼
                         ask:          CRITIC ─► score < 0.7? ─► REVISER ─► CRITIC ... (max 2 rounds,
                                                                                    best-so-far wins)
                         add_example:  CRITIC only (the user's Q&A is never rewritten);
                                       low score -> issues -> human approval
                                               ▼
                                     policy.authorize (code always has the last word)
```

- **Correctable** (fed back to the model): malformed JSON / schema errors, empty answer, add params
  not copied verbatim, an example id the research never found, an unexplained intent change.
- **Final** (no retry, policy decides): role denials, duplicates, eval-set leakage, unsupported intents.
- Feedback texts are written by code, never copied from user or tool text; untrusted text in
  prompts is delimited and its tags neutralised.
- Reflection is a quality gate, not a single point of failure: if the critic is down, answers
  proceed unchanged and the policy still decides.
- Config: `ops_crew/config/refine.yaml` (threshold, rounds, retries, a profile per role:
  repair / critic / reviser). `OPS_REFINE_PROFILE_CRITIC=openai` puts a stronger model on the
  critic; `OPS_REFINE=false` switches the whole layer off.
- Local models work: profiles with a `base_url` (Ollama, vLLM, llama.cpp) default to
  `output_mode: prompted` (schema in the prompt, no tool calling needed); set `tool` or
  `native` per profile in `llms.yaml` if your server supports it.
- PydanticAI is pinned to 1.107.7 for compatibility with CrewAI 1.15, and Anthropic profiles
  aren't available to the refiner; see `requirements-crew.txt` for the reasons.

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
