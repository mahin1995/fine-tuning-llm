"""Self-correction: typed output + feedback-driven retries.

Unlike a blind retry (same prompt again, which at temperature 0 tends to repeat the same
mistake), every failed attempt is answered with the exact problem: Pydantic schema errors
are sent back by PydanticAI automatically, and each problem returned by a check is sent as
a ModelRetry. The model sees its own previous output plus the feedback.
"""
from typing import Callable, Sequence, TypeVar

from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.messages import ModelResponse, RetryPromptPart
from pydantic_ai.models import Model
from pydantic_ai.usage import UsageLimits

from refiner.models import OutputMode, output_spec
from refiner.schemas import CorrectionResult, RefinerError

T = TypeVar("T", bound=BaseModel)

# A check returns problems in plain language ([] = OK). Problems are shown to the model,
# so write them as instructions it can act on ("answer is empty; write the answer").
Check = Callable[[T], list[str]]


def format_problems(problems: list[str]) -> str:
    return "Your previous output has these problems. Fix all of them and keep everything else unchanged:\n" + \
        "\n".join(f"- {p}" for p in problems)


def self_correct(
    model: Model,
    output_type: type[T],
    prompt: str,
    *,
    instructions: str | None = None,
    checks: Sequence[Check] = (),
    max_retries: int = 2,
    output_mode: OutputMode = "tool",
) -> CorrectionResult[T]:
    """Get an output that passes the schema and every check, retrying with feedback.

    Raises RefinerError when the retry budget is exhausted or the provider fails.
    """
    agent = Agent(model, output_type=output_spec(output_type, output_mode), instructions=instructions,
                  retries={"output": max_retries})

    @agent.output_validator
    def run_checks(output: T) -> T:
        problems = [problem for check in checks for problem in check(output)]
        if problems:
            raise ModelRetry(format_problems(problems))
        return output

    try:
        # +1 request for the first attempt; the limit is a hard stop against runaway loops.
        result = agent.run_sync(prompt, usage_limits=UsageLimits(request_limit=max_retries + 1))
    except Exception as e:  # UnexpectedModelBehavior (budget spent), UsageLimitExceeded, provider errors
        raise RefinerError(f"{type(e).__name__}: {e}") from e

    messages = result.all_messages()
    feedback = [part.model_response() for m in messages for part in m.parts if isinstance(part, RetryPromptPart)]
    attempts = sum(isinstance(m, ModelResponse) for m in messages)
    return CorrectionResult(output=result.output, attempts=attempts, feedback=feedback)
