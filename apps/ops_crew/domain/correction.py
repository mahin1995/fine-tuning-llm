"""Which problems an LLM may be asked to fix (correctable), and the rules for reflection results.

Correctable problems are the model's own mistakes: malformed output, a missing answer,
a parameter that wasn't copied verbatim, an id no dataset tool returned. Feeding them
back gives the model a fair chance to fix them.

Everything else is final and decided by policy.py without asking the model again: role
denials, duplicates, eval-set leakage, unsupported intents. Retrying those would only
waste calls, or worse, invite the model to argue its way around a rule.

All problem texts are written by this module (no user or tool text inside), so they are
safe to send back to the model.
"""
from ops_crew.domain.ports import ReflectionOutcome
from ops_crew.domain.working_memory import VerifiedFacts
from ops_crew.schemas import ActionProposal, ClassifiedRequest, Intent

REFLECTED_INTENTS = {Intent.ASK, Intent.ADD_EXAMPLE}


def normalize(text: str) -> str:
    """Whitespace- and case-insensitive form used for every "copied verbatim" check."""
    return " ".join(text.split()).lower()


def correctable_problems(proposal: ActionProposal, classification: ClassifiedRequest, request: str,
                         facts: VerifiedFacts) -> list[str]:
    """`facts`: what the tools returned in this request. An id counts as found only if a tool
    returned it; what the researcher wrote in its findings is a claim, not a fact."""
    problems = []
    if proposal.intent != classification.intent and not proposal.issues:
        problems.append(f"intent '{proposal.intent.value}' differs from the classification "
                        f"'{classification.intent.value}'; use the same intent or explain the change in `issues`")
    if proposal.intent == Intent.ASK and not (proposal.answer and proposal.answer.strip()):
        problems.append("intent is 'ask' but `answer` is empty; write the answer using the research")
    if proposal.intent == Intent.ADD_EXAMPLE:
        source = normalize(request)
        for name in ("question", "answer"):
            value = getattr(proposal.params, name)
            if value and normalize(value) not in source:
                problems.append(f"params.{name} must be copied verbatim from the request, not rewritten")
    if proposal.intent == Intent.REMOVE_EXAMPLE and proposal.params.example_id:
        if proposal.params.example_id not in facts.example_ids:
            options = ", ".join(sorted(facts.example_ids)) or "none"
            problems.append(f"params.example_id {proposal.params.example_id} was not returned by any dataset tool "
                            f"(returned: {options}); use one of those, or leave it empty and set clarifying_question")
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
