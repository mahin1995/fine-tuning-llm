"""Reflection: critique -> revise -> critique, keeping the best version.

Safeguards against the usual failure modes of self-critique:
- flip-flopping (a revision that is worse than an earlier one): the best-scoring version wins
- loops: stop when the reviser returns the same output again
- diminishing returns: stop when the score does not improve
- ungrounded drafts are never accepted (Critique.effective_score)
- a failing critic or reviser ends the loop with the best version so far instead of raising
"""
from typing import Callable, Sequence, TypeVar

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.usage import UsageLimits

from refiner.correction import Check, self_correct
from refiner.models import OutputMode, output_spec
from refiner.schemas import Critique, ReflectionResult, ReflectionRound, RefinerError

T = TypeVar("T", bound=BaseModel)

Critic = Callable[[T], Critique]
Reviser = Callable[[T, Critique], T]


def reflect(draft: T, *, critique: Critic, revise: Reviser, threshold: float = 0.7,
            max_rounds: int = 2) -> ReflectionResult[T]:
    if max_rounds < 0:
        raise ValueError("max_rounds must be >= 0")
    try:
        verdict = critique(draft)
    except RefinerError:
        return ReflectionResult(draft, None, False, "critic_failed")

    current, score = draft, verdict.effective_score()
    rounds = [ReflectionRound(0, draft, score, verdict.issues)]
    best, best_score = draft, score
    if score >= threshold:
        return ReflectionResult(best, best_score, True, "accepted", rounds)

    for n in range(1, max_rounds + 1):
        try:
            revised = revise(current, verdict)
        except RefinerError:
            return ReflectionResult(best, best_score, False, "reviser_failed", rounds)
        if revised.model_dump() == current.model_dump():
            return ReflectionResult(best, best_score, False, "repeated_output", rounds)
        try:
            verdict = critique(revised)
        except RefinerError:
            return ReflectionResult(best, best_score, False, "critic_failed", rounds)
        new_score = verdict.effective_score()
        rounds.append(ReflectionRound(n, revised, new_score, verdict.issues))
        if new_score > best_score:
            best, best_score = revised, new_score
        if new_score >= threshold:
            return ReflectionResult(best, best_score, True, "accepted", rounds)
        if new_score <= score:
            return ReflectionResult(best, best_score, False, "no_improvement", rounds)
        current, score = revised, new_score
    return ReflectionResult(best, best_score, best_score >= threshold, "max_rounds", rounds)


def make_critic(model: Model, *, instructions: str, render: Callable[[T], str],
                output_mode: OutputMode = "tool", max_retries: int = 2) -> Critic:
    """Build a critic: `render(draft)` produces the prompt (context + draft) for the model."""
    agent = Agent(model, output_type=output_spec(Critique, output_mode), instructions=instructions,
                  retries={"output": max_retries})

    def critique(draft: T) -> Critique:
        try:
            return agent.run_sync(render(draft), usage_limits=UsageLimits(request_limit=max_retries + 1)).output
        except Exception as e:
            raise RefinerError(f"critic failed: {type(e).__name__}: {e}") from e

    return critique


def make_reviser(model: Model, output_type: type[T], *, instructions: str, render: Callable[[T, Critique], str],
                 checks: Sequence[Check] = (), output_mode: OutputMode = "tool", max_retries: int = 2) -> Reviser:
    """Build a reviser: it rewrites the draft to address the critique, self-correcting against `checks`."""

    def revise(draft: T, verdict: Critique) -> T:
        return self_correct(model, output_type, render(draft, verdict), instructions=instructions,
                            checks=checks, max_retries=max_retries, output_mode=output_mode).output

    return revise
