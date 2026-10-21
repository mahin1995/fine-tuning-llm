"""Self-correction and reflection for typed LLM outputs, built on PydanticAI.

Generic and domain-agnostic: it knows nothing about datasets, roles or training data.
Callers pass a Pydantic output type, check functions and prompts.

    self_correct(model, OutputType, prompt, checks=[...])
        Ask for a typed output. Schema errors and every problem a check returns are sent
        back to the model as feedback (PydanticAI ModelRetry) until the output is clean
        or the retry budget is spent -> CorrectionResult, or RefinerError.

    reflect(draft, critique=..., revise=...)
        Critic scores the draft; a reviser improves it using the critique; repeat until the
        score passes, rounds run out, the score stops improving or the output repeats.
        Returns the best-scoring version (never a worse one) -> ReflectionResult.

    models: ModelSpec -> PydanticAI model (OpenAI-compatible / Ollama / your own function),
            output modes tool | native | prompted, provider fallback.

Depends only on pydantic and pydantic_ai (enforced by tests/test_architecture.py).
"""
from refiner.correction import Check, self_correct
from refiner.models import ModelSpec, build_model, output_spec, text_function_model
from refiner.reflection import make_critic, make_reviser, reflect
from refiner.schemas import CorrectionResult, Critique, ReflectionResult, ReflectionRound, RefinerError

__all__ = [
    "Check", "CorrectionResult", "Critique", "ModelSpec", "ReflectionResult", "ReflectionRound", "RefinerError",
    "build_model", "make_critic", "make_reviser", "output_spec", "reflect", "self_correct", "text_function_model",
]
