"""What the deterministic flow needs from the probabilistic layers (planner, crew, refiner)."""
from dataclasses import dataclass, field
from typing import Protocol

from ops_crew.domain.working_memory import VerifiedFacts
from ops_crew.schemas import ActionProposal, ClassifiedRequest, CrewRun, Plan, ResearchFindings


class ProposerError(RuntimeError):
    """The crew could not produce output: provider failure, timeout, conversion failure, ..."""


class Proposer(Protocol):
    def propose(self, request: str, correlation_id: str) -> CrewRun:
        """Run the crew on one request. Must not have side effects. Raises ProposerError."""
        ...


@dataclass(frozen=True)
class SimilarExample:
    example_id: str
    score: float  # cosine similarity of the embedder that produced it


@dataclass(frozen=True)
class SimilarityReport:
    """How close a question is to the training and eval data, with the thresholds calibrated for
    the embedder that produced the scores (they differ per embedder). policy.py decides."""
    closest_example: SimilarExample | None
    eval_score: float
    near_duplicate_threshold: float
    near_eval_threshold: float
    embedder: str
    error: str | None = None  # the check could not run: policy asks a human instead

    @classmethod
    def unavailable(cls, error: str) -> "SimilarityReport":
        return cls(None, 0.0, 1.0, 1.0, "none", error)


class SimilarityIndex(Protocol):
    """Meaning-based lookups over the datasets (ops_crew/memory_adapters.py). Eval questions are
    only ever scored, never returned."""

    def search(self, query: str, k: int) -> list[str]:
        """Example ids for a search query, best first (keywords and meaning)."""
        ...

    def similar_examples(self, text: str, k: int = 3) -> list[SimilarExample]:
        ...

    def report(self, text: str) -> SimilarityReport:
        ...


class PlannerError(RuntimeError):
    """The planner could not produce output: provider failure, timeout, ..."""


class Planner(Protocol):
    def plan(self, request: str, correlation_id: str) -> str:
        """Split one request into steps. Returns the raw output; the flow validates it against
        schemas.Plan and domain/planning.py. Must not have side effects. Raises PlannerError."""
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
    facts: VerifiedFacts = VerifiedFacts()  # what the tools returned (working_memory.py)


@dataclass(frozen=True)
class PlanRepairRequest:
    request: str                      # cleaned user request (untrusted)
    previous_output: str              # the planner's raw or re-serialised plan (untrusted)
    problems: list[str]               # written by our code (domain/planning.py, validation): safe to show


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

    def repair_plan(self, req: PlanRepairRequest) -> Plan:
        """Fix a plan's structural problems (domain/planning.plan_problems). Raises RefinementError."""
        ...

    def review_answer(self, request: str, proposal: ActionProposal, findings: ResearchFindings) -> ReflectionOutcome:
        """intent=ask: critique the answer and revise it until it is good enough."""
        ...

    def assess_example(self, request: str, proposal: ActionProposal, findings: ResearchFindings) -> ReflectionOutcome:
        """intent=add_example: critique only. The user's Q&A is never rewritten."""
        ...
