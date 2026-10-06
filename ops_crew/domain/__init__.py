"""Deterministic layer: plain Python, no CrewAI / LLM imports (enforced by tests/test_architecture.py).

    ports        Proposer protocol (what the flow needs from the crew) + ProposerError
    repository   DatasetRepository over data/train.jsonl, reusing qwen_ft.data rules
    inputs       request sanitising + prompt-injection markers
    validation   crew output -> Pydantic models (fence/JSON extraction, exact errors)
    policy       authorization + business rules -> Decision
    idempotency  idempotency keys and stores
    approval     ApprovalGateway for destructive / risky actions
    audit        structured JSON audit log with correlation ids
    actions      ActionExecutor: the only code that changes data
"""
