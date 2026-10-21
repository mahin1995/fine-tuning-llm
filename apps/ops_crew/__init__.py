"""Dataset Ops Assistant: a hybrid agent built on CrewAI.

    LLM proposes  ->  deterministic code validates and decides  ->  deterministic code executes

Layers (dependency direction: crew -> domain, flow -> domain + ports; domain imports neither):

    schemas.py   Pydantic contracts between the layers (crew output, params, results)
    settings.py  configuration from env (OPS_*), no secrets in code
    domain/      deterministic layer, plain Python: validation, policy (authorization +
                 business rules), idempotency, approval, audit log, dataset repository, executor
    flow.py      CrewAI Flow wiring the fixed pipeline: intake -> crew -> validate ->
                 authorize -> (approve) -> execute -> audit
    crew/        probabilistic layer: 3-agent sequential crew, read-only tools, multi-LLM profiles
    evals/       golden cases + runner for the crew (needs a real LLM)

Run: python -m ops_crew --role editor "add this Q&A to the training data: ..."
"""
