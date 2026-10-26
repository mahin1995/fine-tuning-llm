# ROADMAP: Project Goal, Features ও AI Agent Learning Levels

এই file-এ পুরো project-এর এক জায়গায় সারসংক্ষেপ আছে: লক্ষ্য কী, কী বানানো হয়েছে, কোন learning
level-এর কোন অংশ কোথায় implement করা, আর কী বাকি। খুঁটিনাটি design decision-এর জন্য
[PROCESS.md](../PROCESS.md), আর command ও folder layout-এর জন্য [README.md](../README.md) দেখো।

> **শেষ আপডেট:** Level 1 আর Level 2-এর সব অংশ implement হয়েছে (Level 2: context budget, hybrid retrieval, shared working state, episodic memory, consolidation)। Level 3–5-এর কিছু অংশ আগেই হয়ে গেছে।
> Architecture phase 1–3 শেষ (`finetune` core, `apps/`, model profile, `TrainingEngine` port, `model_serving`)।
> **সতর্কতা:** সব code offline-এ test করা হয়েছে (৪৭৮টা test), কিন্তু **আসল LLM বা GPU দিয়ে এখনও চালানো হয়নি।**

---

## ১. Project Goal

একটা ছোট LLM (**Qwen3-0.6B**) Java / Spring Boot interview Q&A দিয়ে fine-tune করা, আর তার
চারপাশে production-মানের **AI agent system** বানানো। এর মাধ্যমে AI agent engineering-এর
৫টা level হাতে-কলমে শেখা।

| লক্ষ্য | মানে |
|---|---|
| Fine-tuning শেখা | Data → training (full / LoRA) → evaluation → serving, একটা RTX 3060 (12GB)-এ |
| Agent engineering শেখা | Agent loop থেকে multi-agent, memory, security আর evaluation পর্যন্ত |
| Production-মানের কাজ | Clean architecture, loose coupling, test, guardrail, audit, idempotency |

---

## ২. Project-এর Package আর Feature

```
finetune/            fine-tuning harness (core)            python -m finetune <command>
model_serving/       model download + OpenAI-compatible API  python -m model_serving <command>
apps/agent/          tool-calling agent loop               python -m agent
apps/ops_crew/       hybrid CrewAI agent (Dataset Ops)     python -m ops_crew "<request>"
apps/refiner/        self-correction + reflection          (ops_crew/refinement.py থেকে ব্যবহার হয়)
apps/memory/         memory toolkit (stdlib only)          (agent আর ops_crew/memory_adapters.py থেকে)
data/                train.jsonl (10), eval.jsonl (8)
tests/               ৪৭৮টা offline test + architecture rule
```

### `finetune/`: Fine-tuning harness
| Feature | কোথায় |
|---|---|
| Data validation (format, duplicate, train/eval leakage, token length) | `finetune/data/`, `finetune/cli/validate_data.py` |
| Training: full FT (fp32 + bf16 autocast + 8-bit AdamW) বা LoRA | `finetune/training/` |
| Training আর inference-এ একই prompt format (`enable_thinking=False`) | `finetune/model_profiles.yaml` → model profile (`profiles.py`) |
| Evaluation: base বনাম fine-tuned (answer loss + পাশাপাশি উত্তর) | `finetune/evaluation/` |
| Terminal chat (streaming) | `finetune/cli/chat.py` |
| FastAPI `/chat` + browser UI | `finetune/serving/` |
| Training engine port (`--engine trl`, পরে Unsloth), `run_info.json`-এ engine ও profile | `finetune/training/engine.py`, `trl_engine.py` |
| Docker (pinned version, torch-এর CUDA build সুরক্ষিত) | `Dockerfile`, `run.sh`, `requirements.txt` |

### `model_serving/`: Model download + serving (আলাদা component, আলাদা image)
| Feature | কোথায় |
|---|---|
| Model registry (source, chat_template_kwargs, max_tokens_limit) | `model_serving/models.yaml`, `src/model_serving/config.py` |
| Hugging Face থেকে download বা local fine-tuned output | `src/model_serving/download.py` |
| OpenAI-compatible API: streaming, tool calling, auth, `max_tokens` cap | `src/model_serving/api/` |
| Engine port: transformers (full / LoRA), fake (test) | `src/model_serving/engines/` |
| Contract check (নিজের server, vLLM, Ollama) + vLLM / Ollama-তে যাওয়ার নির্দেশ | `check.py`, `backends.py` |
| নিজস্ব Docker image | `model_serving/Dockerfile`, `run.sh`, `requirements.txt` |

### `agent/`: Tool-calling agent loop (model-agnostic)
| Feature | কোথায় |
|---|---|
| Loop: model → `<tool_call>` → tool → ফল → আবার model | `agent/loop.py` |
| Tool registry (type hint থেকে JSON schema, argument validation) | `agent/tools.py` |
| Parser (ভাঙা JSON ধরে, শুধু tag দেওয়া call-কেই tool call ধরে) | `agent/parser.py` |
| Built-in tool: নিরাপদ AST calculator, time, knowledge base search | `agent/builtin_tools.py` |
| Qwen backend (একমাত্র যেটা `finetune` import করে) | `agent/backends/qwen.py` |

### `ops_crew/`: Hybrid agent (Dataset Ops Assistant)
**নীতি: LLM প্রস্তাব দেয় → code যাচাই করে সিদ্ধান্ত নেয় → code execute করে।**

| Feature | কোথায় |
|---|---|
| CrewAI Flow: intake → crew → validate → reflect → authorize → approve → execute → audit | `ops_crew/flow.py` |
| ৩-agent sequential crew (Classifier, Researcher, Reviewer) | `ops_crew/crew/`, `config/agents.yaml`, `tasks.yaml` |
| Planner: একাধিক কাজের request-কে ধাপে ভাগ করা (plan-and-execute) | `config/planner.yaml`, `crew/planner.py`, `flow.py` (`PlanFlow`) |
| Plan-এর নিয়ম: হুবহু excerpt, ধাপের নির্ভরতা, plan gate (role, ধাপের সীমা, পুরো plan-এর approval) | `ops_crew/domain/planning.py` |
| Multi-LLM profile (local, OpenAI, Anthropic, finetuned), প্রতিটা agent-এ আলাদা, fallback সহ | `ops_crew/crew/llm.py`, `config/llms.yaml` |
| Read-only, traced tool | `ops_crew/crew/tools.py` |
| Role ও business rule (plain Python) | `ops_crew/domain/policy.py` |
| Idempotency key, atomic write | `ops_crew/domain/idempotency.py`, `actions.py`, `repository.py` |
| Human approval | `ops_crew/domain/approval.py` |
| Correlation id সহ JSON audit log | `ops_crew/domain/audit.py` |
| ২০টা golden eval case (৬টা planning-এর) + runner | `ops_crew/evals/` |

### `refiner/`: Self-correction ও Reflection (PydanticAI, domain-agnostic)
| Feature | কোথায় |
|---|---|
| `self_correct()`: schema বা check-এর ভুল model-কে feedback হিসেবে ফেরত | `refiner/correction.py` |
| `reflect()`: critic → reviser loop, best-so-far, থামার নিয়ম | `refiner/reflection.py` |
| Local model (Ollama, vLLM, in-process), output mode tool / native / prompted, fallback | `refiner/models.py` |
| `ops_crew`-এর সাথে সংযোগ (একমাত্র bridge) | `ops_crew/refinement.py`, `config/refine.yaml` |

### `memory/`: Memory toolkit (Level 2, stdlib only, domain-agnostic)
| Feature | কোথায় |
|---|---|
| BM25 + embedding (offline hashing বা যেকোনো OpenAI-compatible `/embeddings`), Reciprocal Rank Fusion, provider fail করলে hashing-এ ফেরা | `memory/retrieval.py` |
| Token budget আর deterministic history compaction (tool call আর তার ফল কখনো আলাদা হয় না) | `memory/context.py` |
| SQLite episode store (request, decline, fact) আর retention | `memory/episodes.py` |
| একই রকম text group করা (consolidation-এর জন্য) | `memory/consolidation.py` |
| Agent: context budget, hybrid knowledge base, opt-in user memory (`remember` শুধু user-এর হুবহু কথা) | `agent/loop.py`, `agent/builtin_tools.py`, `agent/user_memory.py` |
| ops_crew: near-duplicate approval, `find_similar`, `verified_facts`, `--reply-to`, decline memory, consolidation | `ops_crew/memory_adapters.py`, `domain/working_memory.py`, `domain/followup.py`, `consolidate.py` |

---

## ৩. Learning Levels: কী হয়েছে, কোথায়, কী বাকি

চিহ্ন: ✅ সম্পূর্ণ · ⚠️ আংশিক · ❌ শুরু হয়নি

### LEVEL 1: Core Reasoning & AI Agent Loops
| বিষয় | অবস্থা | কোথায় | বাকি |
|---|---|---|---|
| Agent Loops | ✅ | `agent/loop.py` (step limit, error feedback, repeat-call dedupe, output truncation) | — |
| Deterministic vs Probabilistic Decisions | ✅ | `ops_crew/domain/` (code সিদ্ধান্ত নেয়) বনাম `ops_crew/crew/` (LLM প্রস্তাব দেয়) | — |
| Task Decomposition & Planning | ✅ | Plan-and-execute: planner (`config/planner.yaml`, `crew/planner.py`) typed `Plan` দেয়; `domain/planning.py` structure যাচাই করে আর plan gate চালায়; `PlanFlow` (`flow.py`) প্রতিটা ধাপ পুরো pipeline-এর ভেতর দিয়ে চালায়; plan ভাঙা হলে refiner দিয়ে repair | আসল LLM-এ মাপা; ধাপের মধ্যে data পাঠানো (ReWOO); planned intent crew-কে hint হিসেবে দেওয়া |
| Self-Correction & Reflection | ✅ | `refiner/`, `ops_crew/domain/correction.py`, `flow.py` (`reflect_on_proposal`) | আসল LLM-এ মাপা |

**Level 1 অগ্রগতি: ~১০০%** (সব অংশ implement আর offline-এ test করা; আসল LLM-এ মাপা বাকি)

### LEVEL 2: Context Engineering & AI Agent Memory
| বিষয় | অবস্থা | কোথায় | বাকি |
|---|---|---|---|
| Context Engineering | ✅ | Untrusted text-এর delimiter; agent-এ token budget আর deterministic compaction (`memory/context.py`, `agent/loop.py`, `--context-budget`); Qwen tokenizer দিয়ে exact token গোনা; recall হওয়া memory budget মেনে | আসল LLM-এ budget-এর প্রভাব মাপা |
| Episodic Memory | ✅ | প্রতিটা request আর human decline SQLite-এ (`ops_crew/memory_adapters.py`); `--reply-to` দিয়ে clarification-এর পরে কাজ চালিয়ে যাওয়া; decline হওয়া action আবার human-এর কাছে যায় | Service হলে user অনুযায়ী আলাদা করা |
| Semantic Memory | ✅ | BM25 + embedding hybrid retrieval (`memory/retrieval.py`); agent knowledge base; ops_crew search, `find_similar`, প্রায়-duplicate বা প্রায়-eval হলে approval; agent-এর opt-in user memory | আসল embedding model দিয়ে threshold calibrate করা |
| Shared Working State | ✅ | Tool যা ফেরত দিয়েছে তা code রাখে (`ToolCallRecord.facts`, `domain/working_memory.py`); reviewer পড়ে `verified_facts` দিয়ে; correction শুধু tool-এর ফেরত দেওয়া id মানে | — |
| Memory Consolidation | ✅ | `python -m ops_crew.consolidate`: report, golden candidate, training candidate (human যাচাই করে), `--prune` দিয়ে retention | নিয়মিত চালানো (cron) আর ফলাফল PROCESS.md-তে লিখে রাখা |

**Level 2 অগ্রগতি: ~১০০%** (সব অংশ implement আর offline-এ test করা; আসল LLM আর embedding model-এ মাপা বাকি)

### LEVEL 3: Tools, Execution & Human Control
| বিষয় | অবস্থা | কোথায় | বাকি |
|---|---|---|---|
| Sandbox Execution | ⚠️ | নিরাপদ AST calculator (`eval` নেই), read-only dataset view | আসল sandbox (container / subprocess-এ resource limit), code চালানোর tool |
| Dynamic Tool Retrieval | ❌ | Tool এখন static তালিকা (`agents.yaml`) | অনেক tool থেকে প্রাসঙ্গিকগুলো প্রয়োজনমতো বাছাই |
| Human-in-the-Loop | ✅ | `ops_crew/domain/approval.py`, flow-এর `request_approval` | Async approval (queue বা web UI) |
| Idempotency & Safe Retries | ✅ | `ops_crew/domain/idempotency.py`, retry বাজেট, transient বনাম permanent error | — |

**Level 3 অগ্রগতি: ~৫৫%**

### LEVEL 4: Multi-Agent Systems & Agent Orchestration
| বিষয় | অবস্থা | কোথায় | বাকি |
|---|---|---|---|
| Orchestrator Agents | ⚠️ | Deterministic orchestrator (CrewAI Flow); `PlanFlow`: LLM planner ধাপ ভাগ করে, code সেগুলো চালায় | LLM-চালিত orchestrator যে কাজ ভাগ করে আলাদা agent-দের দেয় |
| Agent-to-Agent Communication | ⚠️ | Task context দিয়ে sequential handoff | Message-ভিত্তিক বা A2A protocol |
| Adversarial Debate | ❌ | — | দুই agent বিপরীত পক্ষে, judge সিদ্ধান্ত দেয় (যেমন: example-টা ভালো কিনা) |
| Critic Agents | ✅ | Reviewer agent (`agents.yaml`), refiner critic (`refiner/reflection.py`) | — |
| Meta-Agents | ❌ | — | যে agent অন্য agent-এর prompt বা config উন্নত করে (eval ফলাফল দেখে) |
| Dynamic Agent Creation | ❌ | — | কাজ অনুযায়ী runtime-এ agent বানানো |

**Level 4 অগ্রগতি: ~২৫%**

### LEVEL 5: Production AI Agents, Security & Evaluation
| বিষয় | অবস্থা | কোথায় | বাকি |
|---|---|---|---|
| Deterministic Guardrails | ✅ | `policy.py`, `validation.py`, `inputs.py`, Pydantic schema | — |
| AI Agent Evaluation | ✅ | Golden set + oracle test (`ops_crew/evals/`), `finetune/evaluation/` | আসল LLM-এ pass rate, regression track করা |
| Prompt Injection Defense | ✅ | Delimiter + tag neutralise, read-only tool, role আসে caller থেকে, injection marker থাকলে approval | Red-team eval বাড়ানো |
| AI Agent Security | ⚠️ | Least privilege (role), secret শুধু env-এ, architecture boundary test | Rate limit, PII redaction, secret scanning, audit log-এর integrity |
| Distributed Tracing | ⚠️ | Correlation id সহ audit log | OpenTelemetry span (flow → crew → LLM → tool) |
| Agent Telemetry | ❌ | Telemetry বন্ধ রাখা (`CREWAI_DISABLE_TELEMETRY`) | Latency, token, খরচ, retry rate-এর metric আর dashboard |

**Level 5 অগ্রগতি: ~৫৫%**

---

## ৪. Architecture-এর নিয়ম (test দিয়ে পাহারা দেওয়া)

| নিয়ম | Test |
|---|---|
| `finetune`: config ← data/modeling ← training/evaluation/serving ← cli | `tests/test_architecture.py` |
| `agent` core `finetune` বা ML library import করে না | একই |
| `ops_crew/domain/` plain Python (CrewAI নেই) | একই |
| `refiner` শুধু pydantic / pydantic_ai import করে | একই |
| `finetune`, `agent` আর `ops_crew` একে অপরকে import করে না | একই |
| `model_serving` project-এর কিছু import করে না, কেউ তাকেও import করে না (শুধু HTTP) | একই |
| Serving আর training-এর `chat_template_kwargs` মেলে | `tests/test_model_serving_integration.py` |
| Plan-এর ধাপ শুধু user-এর হুবহু শব্দ; কোনো side effect-এর আগে plan gate; প্রতিটা ধাপ পুরো pipeline-এর ভেতর দিয়ে | `tests/test_ops_planning.py`, `tests/test_ops_plan_flow.py` |
| `memory` শুধু standard library import করে; ops_crew-এ শুধু `memory_adapters.py` তাকে import করে | `tests/test_architecture.py` |
| Memory poisoning: tool output বা model-এর লেখা memory-তে যায় না; প্রায়-duplicate হলে human | `tests/test_agent_memory.py`, `tests/test_ops_memory*.py` |

---

## ৫. পরের ধাপ (ক্রম অনুযায়ী)

1. **আসল LLM দিয়ে যাচাই** *(সবচেয়ে জরুরি)*
   - `./run.sh build && ./run.sh python -m pytest`
   - Ollama-য় `qwen2.5:7b-instruct`, অথবা কোনো API key দিয়ে: `./run.sh python -m ops_crew.evals --report outputs/ops/evals.json`
   - Pass rate, self-corrected, reflection আর planning-এর সংখ্যা (planned, multi-step, plans repaired) PROCESS.md-তে লিখে রাখা
   - `OPS_PLANNING=false` আর `OPS_MEMORY=false` দিয়ে আলাদা করে চালিয়ে planning আর memory-র লাভ মাপা
   - আসল embedding model (`ollama pull nomic-embed-text`) দিয়ে similarity threshold আবার calibrate করা
2. **Level 3:** sandbox execution, dynamic tool retrieval
3. **Level 4:** LLM orchestrator, adversarial debate (example যাচাই), meta-agent (eval দেখে prompt উন্নত করা)
4. **Level 5:** OpenTelemetry tracing, telemetry metric, security hardening
5. **Fine-tuning-এর লক্ষ্য পূরণ**
   - Dataset ১০ থেকে ৫০০+ example-এ নেওয়া (`ops_crew` দিয়েই curate করা যায়)
   - Distillation: বড় model-এর সফল crew transcript দিয়ে 0.6B-কে LoRA train, তারপর `OPS_LLM_PROFILE=finetuned` দিয়ে eval তুলনা
   - `python -m ops_crew.consolidate`-এর training candidate থেকে human যাচাই করে নতুন example
6. **Architecture phase 4–5:** `apps/chat_ui` (OpenAI API client), agent-এর `OpenAIBackend`, `finetune chat`/`serve` সরানো

---

## ৬. জানা সীমাবদ্ধতা

| সীমাবদ্ধতা | কারণ / সমাধান |
|---|---|
| আসল LLM বা GPU-তে কিছু চালানো হয়নি | Dev container থেকে Hugging Face বা LLM-এ যাওয়া যায় না। নিজের machine-এ চালাতে হবে |
| Dataset মাত্র ১০টা example | Smoke test-এর জন্য যথেষ্ট, আসল training-এর জন্য না |
| 0.6B model multi-agent JSON flow-এ দুর্বল | Crew আর critic-এ 7B+ model ব্যবহার, অথবা distillation |
| Refiner-এ Anthropic নেই | PydanticAI আর CrewAI-র dependency conflict (`apps/requirements.txt`-এ বিস্তারিত) |
| Plan-এর একটা ধাপ শুধু নিজের অংশটুকু দেখে | "এই দুটো যোগ করো: Q1 … Q2 …"-এর মতো shared context-এ একটা ধাপ অস্পষ্ট হতে পারে; তখন crew প্রশ্ন করে বা escalate করে, অনুমানে কিছু করে না। সমাধান: planned intent crew-কে hint হিসেবে দেওয়া |
| Plan-এর মাঝে fail করলে rollback নেই | আগের ধাপ যা করেছে তা থেকে যায়, ফলাফলে `partially_completed` আর কোন ধাপ কী হলো দেখায়। পরে Saga (compensation) |
| Default hashing embedder শুধু শব্দের মিল বোঝে, সমার্থক শব্দ নয় | আসল embedding model (Ollama, vLLM, OpenAI) config দিলে অর্থের মিল ধরা পড়ে; তখন threshold আবার calibrate করতে হবে |
| Memory user অনুযায়ী আলাদা নয় | CLI-তে শুধু role আছে, user identity নেই। Service হলে episode আর user memory user অনুযায়ী আলাদা করতে হবে |
| PydanticAI 1.107.7-এ আটকানো | CrewAI 1.15-এর সাথে চলে এমন সর্বশেষ version। Upgrade-এর আগে দুটো একসাথে import হয় কিনা দেখতে হবে |

---

## ৭. দ্রুত যাচাইয়ের command

```bash
./run.sh python -m pytest                        # সব test (CPU, network লাগে না)
./run.sh python -m finetune validate              # dataset যাচাই
./run.sh python -m finetune --help                # fine-tuning command
./run.sh python -m agent --help                  # agent loop
./run.sh python -m ops_crew --help               # hybrid agent
./run.sh python -m ops_crew.evals --help         # eval
model_serving/run.sh serve --model qwen3-0.6b    # OpenAI-compatible server (127.0.0.1:8001)
model_serving/run.sh check --model qwen3-0.6b    # contract check
```
