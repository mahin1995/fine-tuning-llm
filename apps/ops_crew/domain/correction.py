"""Which problems an LLM may be asked to fix (correctable), and the rules for reflection results.

Correctable problems are the model's own mistakes: malformed output, a missing answer,
a parameter that wasn't copied verbatim, an id the research never found. Feeding them
back gives the model a fair chance to fix them.

Everything else is final and decided by policy.py without asking the model again: role
denials, duplicates, eval-set leakage, unsupported intents. Retrying those would only
waste calls, or worse, invite the model to argue its way around a rule.

All problem texts are written by this module (no user or tool text inside), so they are
safe to send back to the model.
"""
from ops_crew.domain.ports import ReflectionOutcome
from ops_crew.schemas import ActionProposal, ClassifiedRequest, Intent, ResearchFindings

REFLECTED_INTENTS = {Intent.ASK, Intent.ADD_EXAMPLE}


def _norm(text: str) -> str:
    return " ".join(text.split()).lower()


def correctable_problems(proposal: ActionProposal, classification: ClassifiedRequest,
                         findings: ResearchFindings, request: str) -> list[str]:
    problems = []
    if proposal.intent != classification.intent and not proposal.issues:
        problems.append(f"intent '{proposal.intent.value}' differs from the classification "
                        f"'{classification.intent.value}'; use the same intent or explain the change in `issues`")
    if proposal.intent == Intent.ASK and not (proposal.answer and proposal.answer.strip()):
        problems.append("intent is 'ask' but `answer` is empty; write the answer using the research")
    if proposal.intent == Intent.ADD_EXAMPLE:
        source = _norm(request)
        for name in ("question", "answer"):
            value = getattr(proposal.params, name)
            if value and _norm(value) not in source:
                problems.append(f"params.{name} must be copied verbatim from the request, not rewritten")
    if proposal.intent == Intent.REMOVE_EXAMPLE and proposal.params.example_id:
        known = set(findings.related_example_ids) | ({findings.duplicate_of} if findings.duplicate_of else set())
        if proposal.params.example_id not in known:
            options = ", ".join(sorted(known)) or "none"
            problems.append(f"params.example_id {proposal.params.example_id} was not found by the research "
                            f"(found: {options}); use one of those, or leave it empty and set clarifying_question")
    return problems


def reflection_route(intent: Intent, outcome: ReflectionOutcome) -> str:
    """Decide what a reflection result means for the flow.

    ask:          accepted -> proceed with the best answer; critic unavailable -> proceed with the
                  original (reflection is a quality gate, not a single point of failure);
                  otherwise -> escalate (don't answer with a low-quality answer)
    add_example:  never blocks by itself; low quality adds the critique to the proposal's issues,
                  which makes the policy require human approval
    """
    if intent == Intent.ASK:
        if outcome.accepted or outcome.score is None:
            return "proceed"
        return "escalate"
    return "proceed"
