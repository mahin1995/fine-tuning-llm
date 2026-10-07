# Architecture: loosely coupled components

Target design for the repository. Every component can be replaced without touching the
others: the training engine (TRL → Unsloth), the inference runtime (own server → vLLM →
Ollama) and the apps that use the model. [apps/ROADMAP.md](apps/ROADMAP.md) tracks feature progress,
[PROCESS.md](PROCESS.md) the decisions taken so far.

> Status: **phases 1–3 done** (rename `qwen_ft` → `finetune`, apps under `apps/`, model profiles,
> `TrainingEngine` port, self-contained `model_serving/`). The other phases in section 6 follow,
> each with the full test suite green.

---

## 1. Principles

1. **Components talk through contracts, never through each other's internals.** There are
   three contracts (section 3): files, an HTTP API, and Python ports inside a component.
2. **Ports and adapters (hexagonal).** Core logic depends on an interface (port); each
   technology (TRL, Unsloth, transformers, vLLM, CrewAI, ...) is an adapter behind it.
   Swapping technology = adding or choosing an adapter.
3. **One component, one dependency set, one image.** A library upgrade in one component
   can't break another.
4. **Configuration at the edges.** Component code reads typed settings; values come from env
   vars or the component's own YAML. Composition (wiring adapters to ports) only happens in
   each component's CLI / entry point.
5. **Rules are tests.** Every dependency rule below is enforced in `tests/test_architecture.py`.

Spring Boot analogy: each component is a separate microservice. Ports are Java interfaces,
adapters are their `@Component` implementations, and composition is the `@Configuration` class.

---

## 2. Components

```
fine-tuning-llm/
├── finetune/          CORE: data → train → evaluate               (training image)
├── model_serving/    SERVING: download + OpenAI-compatible API   (serving image, self-contained)
├── apps/             APPS: agent, ops_crew, refiner, chat_ui     (apps environment)
├── data/             train.jsonl, eval.jsonl                     (file contract)
├── outputs/          model artifacts, audit logs                 (file contract, not in git)
└── tests/            cross-component + architecture tests
```

| Component | Owns | Does not own | Dependencies |
|---|---|---|---|
| **`finetune`** (core) | dataset rules, training, evaluation (loss needs logits, so in-process) | serving, chat, agents | torch, transformers, trl, peft (`requirements.txt`) |
| **`model_serving`** | model download, inference server with OpenAI API | training, generation policy (`max_tokens`, temperature come from callers) | torch, transformers, fastapi, huggingface_hub (`model_serving/requirements.txt`) |
| **`apps`** | everything that *uses* a model: agent loop, ops crew, refiner, chat UI | how the model is hosted | crewai, pydantic-ai, openai client (`apps/requirements.txt`) |

---

## 3. Contracts

### 3.1 File contract: dataset (`data/*.jsonl`)
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```
- Owner: `finetune.data` (schema + validation).
- Writers: people and `apps/ops_crew` (through `finetune.data` validation only).
- Readers: `finetune` training and evaluation, `apps` tools (read-only).

### 3.2 File contract: model artifact (`outputs/<name>/`)
```
model.safetensors | adapter_model.safetensors + adapter_config.json   (full or LoRA)
tokenizer files + chat template
run_info.json      base model, engine (trl | unsloth), profile + chat_template_kwargs, options,
                   metrics, library versions
```
- Producer: any `TrainingEngine`.
- Consumers: `finetune evaluate`, `model_serving`, vLLM (reads it as is), Ollama (via a GGUF export).

### 3.3 HTTP contract: OpenAI-compatible inference API
```
GET  /health
GET  /v1/models
POST /v1/chat/completions   {model, messages, max_tokens, temperature, top_p, stop, stream,
                             tools?, chat_template_kwargs?}
```
- Implemented by `model_serving`, **vLLM** and **Ollama**. Switching runtime only changes a
  URL (`*_BASE_URL` env var). Nothing else changes.
- Generation policy (`max_tokens`, temperature) is chosen by the caller in every request. The
  server only enforces a hard `max_tokens_limit` to protect the GPU.
- `chat_template_kwargs` (e.g. `enable_thinking: false` for Qwen3) is a per-model default in
  `model_serving` and is sent explicitly by clients when the runtime is vLLM.
- A contract test (`python -m model_serving check --base-url ...`) runs the same checks against
  any runtime before you switch to it.

---

## 4. Ports and adapters per component

### `finetune` (core)
| Port | Adapters | Notes |
|---|---|---|
| `TrainingEngine.train(TrainOptions) -> TrainResult` | `TrlEngine` (today), `UnslothEngine` (future) | same `TrainOptions`, same artifact contract |
| `ModelLoader` (for evaluation) | transformers in-process | evaluation needs per-token loss |

Model-family settings: `model_profiles.yaml` + `profiles.resolve_profile()` (chat template variables,
LoRA target layers), recorded in `run_info.json`. Unchanged: `data/` (schema, validation, split).

### `model_serving`
| Port | Adapters | Notes |
|---|---|---|
| `ModelSource.fetch(spec) -> path` | Hugging Face Hub, local directory | download only, no torch needed |
| `InferenceEngine.generate/stream(request)` | `TransformersEngine` (today), `FakeEngine` (tests) | vLLM and Ollama replace the **whole server**, not this port |

### `apps`
| App | Port it depends on | Adapters |
|---|---|---|
| `agent` | `ChatBackend` | `QwenBackend` (in-process, today), `OpenAIBackend` (HTTP, new) |
| `ops_crew` | LLM profiles (`llms.yaml`), `Proposer`, `Refiner` | CrewAI crew, `refiner` via `refinement.py` |
| `refiner` | PydanticAI `Model` | OpenAI-compatible / Ollama / function model |
| `chat_ui` | OpenAI HTTP API | any runtime from 3.3 |

---

## 5. Dependency rules (enforced by tests)

| From \ May import | `finetune.data` | rest of `finetune` | `model_serving` | `apps/*` |
|---|---|---|---|---|
| `finetune` | ✅ | ✅ (internal layering as today) | ❌ | ❌ |
| `model_serving` | ❌ | ❌ | ✅ | ❌ |
| `apps/ops_crew` | ✅ (dataset rules only) | ❌ | ❌ (HTTP only) | own + `refiner` via `refinement.py` |
| `apps/agent` | ❌ | only `agent/backends/qwen.py` (in-process adapter) | ❌ (HTTP only) | ❌ |
| `apps/refiner` | ❌ | ❌ | ❌ | ❌ (pydantic / pydantic_ai only) |
| `apps/chat_ui` | ❌ | ❌ | ❌ (HTTP only) | ❌ |

Rules carried over unchanged: `ops_crew/domain` stays plain Python, only `ops_crew/crew`, `flow`,
`evals` and `__main__` import CrewAI, and lightweight modules never import torch at module level.

**Duplication accepted on purpose:** `model_serving` re-implements model loading (full or
LoRA) instead of importing `finetune.modeling`, so the serving image never depends on the
training code. A cross-component test checks that `model_serving`'s Qwen3
`chat_template_kwargs` equal the `qwen3` entry of `finetune/model_profiles.yaml`.

---

## 6. Migration phases (each one commit or more, tests green after each)

| Phase | Change | Behaviour change |
|---|---|---|
| 1 ✅ | Rename `qwen_ft` → `finetune` (model-agnostic name); in-process LLM profile `qwen_ft` → `finetuned`. Move `agent/`, `ops_crew/`, `refiner/`, `requirements-crew.txt` (→ `apps/requirements.txt`), `ROADMAP.md` into `apps/`. Add `apps` to PYTHONPATH (`pytest.ini`, `run.sh`). Imports and commands stay the same (`python -m ops_crew`). | none |
| 2 ✅ | `finetune`: model profiles (`config/models.yaml`: chat_template_kwargs, LoRA target modules per model family), extract `TrainingEngine` port, current trainer becomes `TrlEngine`, `--engine trl` (default), `run_info.json` records the engine. | none |
| 3 ✅ | `model_serving/`: config, download, OpenAI API, transformers engine, streaming, tools, contract check, own requirements, Dockerfile, run script. | new component |
| 4 | `apps/chat_ui` as an OpenAI API client (UI moved from `finetune/serving`), `agent` `OpenAIBackend`, `finetune chat`/`serve` deprecated then removed. | chat goes through the server |
| 5 | Architecture tests for the new rules, cross-component consistency tests, docs. | none |
| later | `UnslothEngine`, vLLM / Ollama runtime switch (URL only), GGUF export for Ollama. | adapters only |

---

## 7. What changes when you switch technology

| Switch | What you touch | What stays the same |
|---|---|---|
| TRL → **Unsloth** | add `UnslothEngine` + its requirements; `--engine unsloth` | data, options, evaluation, artifact format, serving, apps |
| own server → **vLLM** | run the vLLM container, set `*_BASE_URL`; clients send `chat_template_kwargs` | training, apps code, chat UI |
| own server → **Ollama** | export GGUF + Modelfile, `ollama create`, set `*_BASE_URL` | training, apps code, chat UI |
| local model → **hosted API** (OpenAI etc.) | `llms.yaml` profile / env | everything else |
