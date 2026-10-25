"""Deterministic layer: plain Python, no CrewAI / LLM imports (enforced by tests/test_architecture.py).

    ports        Planner / Proposer / Refiner protocols (what the flow needs from the LLM layers)
    repository   DatasetRepository over data/train.jsonl, reusing finetune.data rules
    inputs       request sanitising + prompt-injection markers
    validation   crew / planner output -> Pydantic models (fence/JSON extraction, exact errors)
    correction   which proposal problems the model may fix itself, reflection rules
    planning     plan checks, the plan gate (role, size, approval), step dependencies, outcome
    policy       authorization + business rules -> Decision
    idempotency  idempotency keys and stores
    approval     ApprovalGateway for destructive / risky actions
    audit        structured JSON audit log with correlation ids
    actions      ActionExecutor: the only code that changes data
"""
