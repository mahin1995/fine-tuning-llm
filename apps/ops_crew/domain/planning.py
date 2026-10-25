"""Planning rules: what a plan may look like, whether it may run, and what its steps add up to.

The planner (an LLM) only proposes a Plan. This module decides, in code:

    plan_problems   structural mistakes the planner can fix itself (fed back for self-correction)
    plan_gate       role / size / approval decision for the whole plan, before any step runs
    step_input      what each step's pipeline gets (a single step is exactly the unplanned path)
    blocked_by      unfinished dependencies that stop a step
    plan_outcome    one outcome for the whole request from the step outcomes

Every step is a verbatim excerpt of the user's request, so a plan can split and order the
user's words but never add content. Each step then runs through the normal pipeline (crew,
validation, policy, approval, executor), so a plan never bypasses a rule. `depends_on` may only
name earlier steps: plan order is a topological order and a cycle is impossible.

Problem texts are written here; the only values inside them are step ids, which the schema
restricts to s1..s99. They are safe to send back to the model.
"""
import hashlib
import json
from dataclasses import dataclass, field

from ops_crew.domain.correction import normalize
from ops_crew.domain.policy import REQUIRED_ROLE, SIDE_EFFECTS, Decision, Outcome, has_role
from ops_crew.schemas import Intent, Plan, PlanStep, Role

COMPLETED = "completed"
PARTIALLY_COMPLETED = "partially_completed"
SKIPPED = "skipped"
MAX_SUMMARY_EXCERPT_CHARS = 120


@dataclass(frozen=True)
class StepResult:
    step_id: str
    intent: str | None          # planned intent; None when the step was not planned (single step)
    outcome: str                # a flow outcome (completed, rejected, ...) or "skipped"
    message: str
    correlation_id: str | None = None
    data: dict = field(default_factory=dict)


def plan_problems(plan: Plan, request: str) -> list[str]:
    """Correctable mistakes. A single-step plan always passes: its step runs on the whole request."""
    if len(plan.steps) == 1:
        return []
    problems = []
    source = normalize(request)
    excerpts: dict[str, str] = {}
    for index, step in enumerate(plan.steps, start=1):
        if step.id != f"s{index}":
            problems.append(f"step {index} has id {step.id}; number the steps s1, s2, s3 ... in order")
        excerpt = normalize(step.excerpt)
        if not excerpt:
            problems.append(f"step {step.id}: excerpt is empty")
        elif excerpt not in source:
            problems.append(f"step {step.id}: excerpt is not copied verbatim from the request; copy the exact words")
        else:
            for earlier_id, earlier in excerpts.items():
                if excerpt in earlier or earlier in excerpt:
                    problems.append(f"steps {earlier_id} and {step.id} overlap (one excerpt contains the other); "
                                    "give every step its own part of the request")
            excerpts[step.id] = excerpt
        earlier_ids = {s.id for s in plan.steps[: index - 1]}
        if len(set(step.depends_on)) != len(step.depends_on):
            problems.append(f"step {step.id}: depends_on lists a step twice")
        for dep in dict.fromkeys(step.depends_on):
            if dep == step.id:
                problems.append(f"step {step.id} depends on itself")
            elif dep not in earlier_ids:
                problems.append(f"step {step.id} depends on {dep}, which is not an earlier step")
    return problems


def plan_gate(plan: Plan, role: Role, *, max_steps: int, suspicious: bool = False) -> Decision:
    """Decide on the whole plan before any step runs.

    A single step is not gated here: it runs through the regular pipeline exactly like an
    unplanned request (the crew decides the intent, policy.authorize decides the rest).
    For several steps every step must be allowed for the role (all or nothing), and a plan
    that changes data is shown to a human as a whole before its first step runs. Each step
    is still authorized again by policy.authorize, with its concrete parameters.
    """
    steps = plan.steps
    if len(steps) == 1:
        return Decision(Outcome.ALLOW, ["single step: handled by the regular pipeline"])
    if len(steps) > max_steps:
        return Decision(Outcome.DENY, [f"the request contains {len(steps)} actions; at most {max_steps} are "
                                       "allowed in one request, send the rest separately"])
    denied = []
    for step in steps:
        required = REQUIRED_ROLE.get(step.intent)
        if required is None:
            denied.append(f"{step.id}: intent {step.intent.value!r} is not supported")
        elif not has_role(role, required):
            denied.append(f"{step.id}: role {role.value!r} may not {step.intent.value} (requires {required.value})")
    if denied:
        return Decision(Outcome.DENY, denied)
    changing = [s.id for s in steps if s.intent in SIDE_EFFECTS]
    if not changing:
        return Decision(Outcome.ALLOW, ["read-only plan"])
    reasons = [f"plan with {len(steps)} steps changes data ({', '.join(changing)}): approve the whole plan first"]
    if suspicious:
        reasons.append("request contains prompt-injection markers")
    return Decision(Outcome.NEEDS_APPROVAL, reasons)


def plan_hash(plan: Plan) -> str:
    """Identifies exactly what a human approved: the steps (the rationale is not part of it)."""
    payload = json.dumps([s.model_dump(mode="json") for s in plan.steps], sort_keys=True, ensure_ascii=False)
    return "plan-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def describe_plan(plan: Plan) -> str:
    """One line per step, for the approval prompt and the audit log."""
    lines = []
    for step in plan.steps:
        after = f" (after {', '.join(step.depends_on)})" if step.depends_on else ""
        excerpt = " ".join(step.excerpt.split())
        if len(excerpt) > MAX_SUMMARY_EXCERPT_CHARS:
            excerpt = excerpt[: MAX_SUMMARY_EXCERPT_CHARS - 3] + "..."
        lines.append(f"{step.id} {step.intent.value}{after}: {excerpt!r}")
    return "\n".join(lines)


def step_input(plan: Plan, step: PlanStep, request: str) -> tuple[str, Intent | None]:
    """(text for the step's crew, intent the crew must confirm). A single step gets the whole
    request and no expected intent: exactly the unplanned path."""
    if len(plan.steps) == 1:
        return request, None
    return step.excerpt, step.intent


def blocked_by(step: PlanStep, results: dict[str, StepResult]) -> list[str]:
    """Dependencies that did not complete. Steps run in plan order and only depend on earlier
    steps, so every dependency already has a result."""
    return [dep for dep in step.depends_on if dep not in results or results[dep].outcome != COMPLETED]


def plan_outcome(results: list[StepResult]) -> str:
    """completed: every step completed; partially_completed: some did; otherwise the outcome of
    the first step that ran (the first step has no dependencies, so it always runs)."""
    if not results:
        raise ValueError("a plan has at least one step")
    completed = sum(r.outcome == COMPLETED for r in results)
    if completed == len(results):
        return COMPLETED
    if completed:
        return PARTIALLY_COMPLETED
    return next((r.outcome for r in results if r.outcome != SKIPPED), SKIPPED)


def plan_message(results: list[StepResult]) -> str:
    if len(results) == 1:
        return results[0].message
    return "\n".join(f"{r.step_id} [{r.outcome}] {r.message}" for r in results)
