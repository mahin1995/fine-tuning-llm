"""Probabilistic layer: the 3-agent sequential crew. The only package that imports CrewAI's
Agent/Task/Crew/LLM. It proposes; it never executes.

    config    load + validate agents.yaml, tasks.yaml, llms.yaml (Pydantic, cross-checked)
    llm       multi-LLM profiles: per-agent selection, env overrides, provider fallback
    tools     read-only dataset tools with call tracing
    proposer  CrewProposer: builds a fresh crew per request, returns a CrewRun
"""
