"""What the deterministic flow needs from the probabilistic layers (crew, refiner)."""
from dataclasses import dataclass, field
from typing import Protocol

from ops_crew.schemas import ActionProposal, ClassifiedRequest, CrewRun, ResearchFindings


class ProposerError(RuntimeError):
    """The crew could not produce output: provider failure, timeout, conversion failure, ..."""


class Proposer(Protocol):
    def propose(self, request: str, correlation_id: str) -> CrewRun:
        """Run the crew on one request. Must not have side effects. Raises ProposerError."""
        ...


class RefinementError(RuntimeError):
    """Self-correction could not produce a clean proposal (budget spent, provider down, ...)."""


@dataclass(frozen=True)
class RepairRequest:
    request: str                      # cleaned user request (untrusted)
    classification: ClassifiedRequest
    findings: ResearchFindings
    previous_output: str              # the crew's raw or re-serialised proposal (untrusted)
    problems: list[str]               # written by our code: safe to show to the model


@dataclass(frozen=True)
class ReflectionOutcome:
    proposal: ActionProposal          # best version (unchanged if nothing better was found)
    score: float | None               # None: the critic could not run
    accepted: bool
    stop_reason: str
    rounds: int
    issues: list[str] = field(default_factory=list)


class Refiner(Protocol):
    """Self-correction + reflection, implemented outside the domain (ops_crew/refinement.py)."""

    def repair(self, req: RepairRequest) -> ActionProposal:
        """Fix the listed problems with feedback-driven retries. Raises RefinementError."""
        ...

    def review_answer(self, request: str, proposal: ActionProposal, findings: ResearchFindings) -> ReflectionOutcome:
        """intent=ask: critique the answer and revise it until it is good enough."""
        ...

    def assess_example(self, request: str, proposal: ActionProposal, findings: ResearchFindings) -> ReflectionOutcome:
        """intent=add_example: critique only. The user's Q&A is never rewritten."""
        ...
