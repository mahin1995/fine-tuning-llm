# ROADMAP: Project Goal, Features ও AI Agent Learning Levels

এই file-এ পুরো project-এর এক জায়গায় সারসংক্ষেপ আছে: লক্ষ্য কী, কী বানানো হয়েছে, কোন learning
level-এর কোন অংশ কোথায় implement করা, আর কী বাকি। খুঁটিনাটি design decision-এর জন্য
[PROCESS.md](PROCESS.md), আর command ও folder layout-এর জন্য [README.md](README.md) দেখো।

> **শেষ আপডেট:** Level 1 শেষ পর্যায়ে (planning বাকি), Level 3–5-এর কিছু অংশ আগেই হয়ে গেছে।
> **সতর্কতা:** সব code offline-এ test করা হয়েছে (২৫৫টা test), কিন্তু **আসল LLM দিয়ে এখনও চালানো হয়নি।**

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
qwen_ft/    fine-tuning harness                   python -m qwen_ft <command>
agent/      tool-calling agent loop               python -m agent
ops_crew/   hybrid CrewAI agent (Dataset Ops)     python -m ops_crew "<request>"
refiner/    self-correction + reflection          (ops_crew/refinement.py থেকে ব্যবহার হয়)
data/       train.jsonl (10), eval.jsonl (8)
tests/      255টা offline test + architecture rule
```

### `qwen_ft/`: Fine-tuning harness
| Feature | কোথায় |
|---|---|
| Data validation (format, duplicate, train/eval leakage, token length) | `qwen_ft/data/`, `qwen_ft/cli/validate_data.py` |
| Training: full FT (fp32 + bf16 autocast + 8-bit AdamW) বা LoRA | `qwen_ft/training/` |
| Training আর inference-এ একই prompt format (`enable_thinking=False`) | `qwen_ft/config.py` → `CHAT_TEMPLATE_KWARGS` |
| Evaluation: base বনাম fine-tuned (answer loss + পাশাপাশি উত্তর) | `qwen_ft/evaluation/` |
| Terminal chat (streaming) | `qwen_ft/cli/chat.py` |
| FastAPI `/chat` + browser UI | `qwen_ft/serving/` |
| Docker (pinned version, torch-এর CUDA build সুরক্ষিত) | `Dockerfile`, `run.sh`, `requirements.txt` |

### `agent/`: Tool-calling agent loop (model-agnostic)
| Feature | কোথায় |
|---|---|
| Loop: model → `<tool_call>` → tool → ফল → আবার model | `agent/loop.py` |
| Tool registry (type hint থেকে JSON schema, argument validation) | `agent/tools.py` |
| Parser (ভাঙা JSON ধরে, শুধু tag দেওয়া call-কেই tool call ধরে) | `agent/parser.py` |
| Built-in tool: নিরাপদ AST calculator, time, knowledge base search | `agent/builtin_tools.py` |
| Qwen backend (একমাত্র যেটা `qwen_ft` import করে) | `agent/backends/qwen.py` |

### `ops_crew/`: Hybrid agent (Dataset Ops Assistant)
**নীতি: LLM প্রস্তাব দেয় → code যাচাই করে সিদ্ধান্ত নেয় → code execute করে।**

| Feature | কোথায় |
|---|---|
| CrewAI Flow: intake → crew → validate → reflect → authorize → approve → execute → audit | `ops_crew/flow.py` |
| ৩-agent sequential crew (Classifier, Researcher, Reviewer) | `ops_crew/crew/`, `config/agents.yaml`, `tasks.yaml` |
| Multi-LLM profile (local, OpenAI, Anthropic, qwen_ft), প্রতিটা agent-এ আলাদা, fallback সহ | `ops_crew/crew/llm.py`, `config/llms.yaml` |
| Read-only, traced tool | `ops_crew/crew/tools.py` |
| Role ও business rule (plain Python) | `ops_crew/domain/policy.py` |
| Idempotency key, atomic write | `ops_crew/domain/idempotency.py`, `actions.py`, `repository.py` |
| Human approval | `ops_crew/domain/approval.py` |
| Correlation id সহ JSON audit log | `ops_crew/domain/audit.py` |
| ১৪টা golden eval case + runner | `ops_crew/evals/` |

### `refiner/`: Self-correction ও Reflection (PydanticAI, domain-agnostic)
| Feature | কোথায় |
|---|---|
| `self_correct()`: schema বা check-এর ভুল model-কে feedback হিসেবে ফেরত | `refiner/correction.py` |
| `reflect()`: critic → reviser loop, best-so-far, থামার নিয়ম | `refiner/reflection.py` |
| Local model (Ollama, vLLM, in-process), output mode tool / native / prompted, fallback | `refiner/models.py` |
| `ops_crew`-এর সাথে সংযোগ (একমাত্র bridge) | `ops_crew/refinement.py`, `config/refine.yaml` |

---

## ৩. Learning Levels: কী হয়েছে, কোথায়, কী বাকি

চিহ্ন: ✅ সম্পূর্ণ · ⚠️ আংশিক · ❌ শুরু হয়নি

### LEVEL 1: Core Reasoning & AI Agent Loops
| বিষয় | অবস্থা | কোথায় | বাকি |
|---|---|---|---|
| Agent Loops | ✅ | `agent/loop.py` (step limit, error feedback, repeat-call dedupe, output truncation) | — |
| Deterministic vs Probabilistic Decisions | ✅ | `ops_crew/domain/` (code সিদ্ধান্ত নেয়) বনাম `ops_crew/crew/` (LLM প্রস্তাব দেয়) | — |
| Task Decomposition & Planning | ⚠️ | Crew-এর classify → research → review ভাগ **আগে থেকে ঠিক করা** (`tasks.yaml`) | LLM নিজে plan বানাবে, ধাপের নির্ভরতা থাকবে, fail হলে re-planning |
| Self-Correction & Reflection | ✅ | `refiner/`, `ops_crew/domain/correction.py`, `flow.py` (`reflect_on_proposal`) | আসল LLM-এ মাপা |

**Level 1 অগ্রগতি: ~৮৫%**

### LEVEL 2: Context Engineering & AI Agent Memory
| বিষয় | অবস্থা | কোথায় | বাকি |
|---|---|---|---|
| Context Engineering | ⚠️ | Untrusted text-এর জন্য delimiter (`<user_request>`, `<tool_data>`), tool output কেটে ছোট করা, chat history trim (`qwen_ft/cli/chat.py`) | Context budget (token গোনা), relevance অনুযায়ী বাছাই, summarization |
| Episodic Memory | ❌ | — | আগের run (audit log) থেকে "কী ঘটেছিল" মনে রাখা |
| Semantic Memory | ⚠️ | `KnowledgeBase` keyword search (`agent/builtin_tools.py`) | Embedding দিয়ে vector search, তথ্য সংরক্ষণ |
| Shared Working State | ⚠️ | CrewAI Flow state (`OpsState`), task context | Agent-দের মধ্যে typed shared scratchpad |
| Memory Consolidation | ❌ | — | Episode থেকে স্থায়ী শিক্ষা বের করা (যেমন: বারবার fail হওয়া request-এর ধরন) |

**Level 2 অগ্রগতি: ~১৫%**

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
| Orchestrator Agents | ⚠️ | Deterministic orchestrator (CrewAI Flow) | LLM-চালিত orchestrator যে কাজ ভাগ করে agent-দের দেয় |
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
| AI Agent Evaluation | ✅ | Golden set + oracle test (`ops_crew/evals/`), `qwen_ft/evaluation/` | আসল LLM-এ pass rate, regression track করা |
| Prompt Injection Defense | ✅ | Delimiter + tag neutralise, read-only tool, role আসে caller থেকে, injection marker থাকলে approval | Red-team eval বাড়ানো |
| AI Agent Security | ⚠️ | Least privilege (role), secret শুধু env-এ, architecture boundary test | Rate limit, PII redaction, secret scanning, audit log-এর integrity |
| Distributed Tracing | ⚠️ | Correlation id সহ audit log | OpenTelemetry span (flow → crew → LLM → tool) |
| Agent Telemetry | ❌ | Telemetry বন্ধ রাখা (`CREWAI_DISABLE_TELEMETRY`) | Latency, token, খরচ, retry rate-এর metric আর dashboard |

**Level 5 অগ্রগতি: ~৫৫%**

---

## ৪. Architecture-এর নিয়ম (test দিয়ে পাহারা দেওয়া)

| নিয়ম | Test |
|---|---|
| `qwen_ft`: config ← data/modeling ← training/evaluation/serving ← cli | `tests/test_architecture.py` |
| `agent` core `qwen_ft` বা ML library import করে না | একই |
| `ops_crew/domain/` plain Python (CrewAI নেই) | একই |
| `refiner` শুধু pydantic / pydantic_ai import করে | একই |
| `qwen_ft`, `agent` আর `ops_crew` একে অপরকে import করে না | একই |

---

## ৫. পরের ধাপ (ক্রম অনুযায়ী)

1. **আসল LLM দিয়ে যাচাই** *(সবচেয়ে জরুরি)*
   - `./run.sh build && ./run.sh python -m pytest`
   - Ollama-য় `qwen2.5:7b-instruct`, অথবা কোনো API key দিয়ে: `./run.sh python -m ops_crew.evals --report outputs/ops/evals.json`
   - Pass rate, self-corrected আর reflection-এর সংখ্যা PROCESS.md-তে লিখে রাখা
2. **Level 1 শেষ করা: Planning**
   - Planner একটা typed `Plan` বানাবে (ধাপ, নির্ভরতা), code প্রতিটা ধাপ validate আর authorize করবে
   - Fail হলে re-planning, ধাপের সর্বোচ্চ সংখ্যা, destructive ধাপ থাকলে plan-এর শুরুতে approval
3. **Level 2: Memory**
   - Episodic: audit log থেকে আগের run খোঁজা
   - Semantic: embedding দিয়ে knowledge search
   - Consolidation: fail হওয়া case থেকে নতুন golden case বা training data বানানো
4. **Level 3:** sandbox execution, dynamic tool retrieval
5. **Level 4:** LLM orchestrator, adversarial debate (example যাচাই), meta-agent (eval দেখে prompt উন্নত করা)
6. **Level 5:** OpenTelemetry tracing, telemetry metric, security hardening
7. **Fine-tuning-এর লক্ষ্য পূরণ**
   - Dataset ১০ থেকে ৫০০+ example-এ নেওয়া (`ops_crew` দিয়েই curate করা যায়)
   - Distillation: বড় model-এর সফল crew transcript দিয়ে 0.6B-কে LoRA train, তারপর `OPS_LLM_PROFILE=qwen_ft` দিয়ে eval তুলনা

---

## ৬. জানা সীমাবদ্ধতা

| সীমাবদ্ধতা | কারণ / সমাধান |
|---|---|
| আসল LLM বা GPU-তে কিছু চালানো হয়নি | Dev container থেকে Hugging Face বা LLM-এ যাওয়া যায় না। নিজের machine-এ চালাতে হবে |
| Dataset মাত্র ১০টা example | Smoke test-এর জন্য যথেষ্ট, আসল training-এর জন্য না |
| 0.6B model multi-agent JSON flow-এ দুর্বল | Crew আর critic-এ 7B+ model ব্যবহার, অথবা distillation |
| Refiner-এ Anthropic নেই | PydanticAI আর CrewAI-র dependency conflict (`requirements-crew.txt`-এ বিস্তারিত) |
| PydanticAI 1.107.7-এ আটকানো | CrewAI 1.15-এর সাথে চলে এমন সর্বশেষ version। Upgrade-এর আগে দুটো একসাথে import হয় কিনা দেখতে হবে |

---

## ৭. দ্রুত যাচাইয়ের command

```bash
./run.sh python -m pytest                        # সব test (CPU, network লাগে না)
./run.sh python -m qwen_ft validate              # dataset যাচাই
./run.sh python -m qwen_ft --help                # fine-tuning command
./run.sh python -m agent --help                  # agent loop
./run.sh python -m ops_crew --help               # hybrid agent
./run.sh python -m ops_crew.evals --help         # eval
```
