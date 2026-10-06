"""Authorization and business rules. All decisions are made here, in code; prompts never
decide what is allowed."""
from dataclasses import dataclass, field
from enum import StrEnum

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


def authorize(proposal: ActionProposal, role: Role, repo: DatasetRepository, *,
              suspicious_input: bool = False, already_executed: bool = False) -> Decision:
    """`already_executed`: the idempotency store holds a result for this exact action. The role
    is still checked, but data rules (e.g. "duplicate example") are skipped: the action already
    happened, and the executor will replay its stored result instead of running it again."""
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
    if reasons:
        return Decision(Outcome.NEEDS_APPROVAL, reasons)
    return Decision(Outcome.ALLOW, ["all checks passed"])
