"""Authorization and business rules. All decisions are made here, in code; prompts never
decide what is allowed."""
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from ops_crew.domain.ports import SimilarityReport
from ops_crew.domain.repository import DatasetRepository
from ops_crew.schemas import ActionProposal, Intent, Role

_ROLE_RANK = {Role.VIEWER: 0, Role.EDITOR: 1, Role.ADMIN: 2}

# Minimum role per intent. Intents missing here (UNKNOWN) are never allowed.
REQUIRED_ROLE = {
    Intent.ASK: Role.VIEWER,
    Intent.DATASET_STATS: Role.VIEWER,
    Intent.ADD_EXAMPLE: Role.EDITOR,
    Intent.REMOVE_EXAMPLE: Role.ADMIN,
}
SIDE_EFFECTS = {Intent.ADD_EXAMPLE, Intent.REMOVE_EXAMPLE}
DESTRUCTIVE = {Intent.REMOVE_EXAMPLE}

MIN_QUESTION_CHARS = 10
MIN_ANSWER_CHARS = 20


class Outcome(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    NEEDS_APPROVAL = "needs_approval"


@dataclass(frozen=True)
class Decision:
    outcome: Outcome
    reasons: list[str] = field(default_factory=list)


def has_role(role: Role, required: Role) -> bool:
    return _ROLE_RANK[role] >= _ROLE_RANK[required]


def _param_problems(proposal: ActionProposal, repo: DatasetRepository) -> list[str]:
    p = proposal.params
    if proposal.intent == Intent.ASK:
        return [] if proposal.answer and proposal.answer.strip() else ["ask: proposal has no answer"]
    if proposal.intent == Intent.ADD_EXAMPLE:
        problems = []
        if not p.question or len(p.question.strip()) < MIN_QUESTION_CHARS:
            problems.append(f"add_example: question missing or shorter than {MIN_QUESTION_CHARS} chars")
        if not p.answer or len(p.answer.strip()) < MIN_ANSWER_CHARS:
            problems.append(f"add_example: answer missing or shorter than {MIN_ANSWER_CHARS} chars")
        if problems:
            return problems
        duplicate = repo.find_duplicate(p.question)
        if duplicate:
            problems.append(f"add_example: duplicate of existing example {duplicate.example_id}")
        if repo.overlaps_eval_set(p.question):
            problems.append("add_example: question is in the eval set (would leak evaluation data)")
        return problems
    if proposal.intent == Intent.REMOVE_EXAMPLE:
        if not p.example_id:
            return ["remove_example: example_id is required"]
        if repo.get(p.example_id) is None:
            return [f"remove_example: example {p.example_id} does not exist"]  # e.g. a hallucinated id
    return []


def similarity_reasons(report: SimilarityReport) -> list[str]:
    """Near duplicates and near eval questions are fuzzy signals: they ask a human, never reject.
    (Exact duplicates and eval questions are rejected by _param_problems before this runs.)"""
    if report.error:
        return [f"similarity check unavailable ({report.error}); check for near duplicates by hand"]
    reasons = []
    closest = report.closest_example
    if closest is not None and closest.score >= report.near_duplicate_threshold:
        reasons.append(f"similar to existing example {closest.example_id} (score {closest.score:.2f})")
    if report.eval_score >= report.near_eval_threshold:
        reasons.append(f"close to an eval question (score {report.eval_score:.2f}): possible leakage")
    return reasons


def authorize(proposal: ActionProposal, role: Role, repo: DatasetRepository, *,
              suspicious_input: bool = False, already_executed: bool = False,
              similarity: SimilarityReport | None = None, declined_at: datetime | None = None) -> Decision:
    """`already_executed`: the idempotency store holds a result for this exact action. The role
    is still checked, but data rules (e.g. "duplicate example") are skipped: the action already
    happened, and the executor will replay its stored result instead of running it again.
    `similarity`: for add_example, how close the question is to existing and eval questions.
    `declined_at`: episodic memory says a human declined this exact action before."""
    required = REQUIRED_ROLE.get(proposal.intent)
    if required is None:
        return Decision(Outcome.DENY, [f"intent {proposal.intent.value!r} is not supported"])
    if not has_role(role, required):
        return Decision(Outcome.DENY, [f"role {role.value!r} may not {proposal.intent.value} "
                                       f"(requires {required.value})"])
    if already_executed:
        return Decision(Outcome.ALLOW, ["replay of an already executed action"])
    problems = _param_problems(proposal, repo)
    if problems:
        return Decision(Outcome.DENY, problems)

    reasons = []
    if proposal.intent in DESTRUCTIVE:
        reasons.append("destructive action requires human approval")
    if proposal.intent in SIDE_EFFECTS and suspicious_input:
        reasons.append("request contains prompt-injection markers")
    if proposal.intent in SIDE_EFFECTS and proposal.issues:
        reasons.append("reviewer raised issues: " + "; ".join(proposal.issues))
    if proposal.intent == Intent.ADD_EXAMPLE and similarity is not None:
        reasons += similarity_reasons(similarity)
    if proposal.intent in SIDE_EFFECTS and declined_at is not None:
        reasons.append(f"a human declined this exact action on {declined_at:%Y-%m-%d %H:%M} UTC")
    if reasons:
        return Decision(Outcome.NEEDS_APPROVAL, reasons)
    return Decision(Outcome.ALLOW, ["all checks passed"])
