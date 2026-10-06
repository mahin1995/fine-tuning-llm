"""Result types for self-correction and reflection."""
from dataclasses import dataclass, field
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T", bound=BaseModel)


class RefinerError(RuntimeError):
    """The model could not produce an acceptable output (retries exhausted, provider failure, ...)."""


class Critique(BaseModel):
    """What the critic returns for one draft."""
    model_config = ConfigDict(extra="forbid")
    score: float = Field(ge=0.0, le=1.0, description="overall quality, 1 = excellent")
    grounded: bool = Field(description="every claim is supported by the provided context")
    issues: list[str] = Field(default_factory=list, description="concrete problems, most important first")

    def effective_score(self) -> float:
        """Ungrounded drafts are never accepted, however confident the critic is about style."""
        return self.score if self.grounded else min(self.score, 0.5)


@dataclass
class CorrectionResult(Generic[T]):
    output: T
    attempts: int                                       # model responses it took
    feedback: list[str] = field(default_factory=list)   # what was sent back to the model, in order


@dataclass
class ReflectionRound(Generic[T]):
    round: int          # 0 = the original draft
    output: T
    score: float        # effective score
    issues: list[str]


StopReason = Literal["accepted", "max_rounds", "no_improvement", "repeated_output", "reviser_failed",
                     "critic_failed"]


@dataclass
class ReflectionResult(Generic[T]):
    best: T
    best_score: float | None   # None when even the first critique failed
    accepted: bool             # best_score reached the threshold
    stop_reason: StopReason
    rounds: list[ReflectionRound] = field(default_factory=list)
