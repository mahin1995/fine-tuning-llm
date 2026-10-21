"""ActionExecutor: the only code that changes data. It runs only after the policy
(and, if needed, a human) allowed the action, and is idempotent for side effects."""
from dataclasses import dataclass, field

from ops_crew.domain.idempotency import IdempotencyStore, idempotency_key
from ops_crew.domain.policy import SIDE_EFFECTS
from ops_crew.domain.repository import DatasetRepository
from ops_crew.schemas import ActionProposal, Intent


@dataclass(frozen=True)
class ExecutionResult:
    status: str                   # "executed" | "replayed" | "answered"
    message: str
    data: dict = field(default_factory=dict)
    idempotency_key: str | None = None


class ActionExecutor:
    def __init__(self, repo: DatasetRepository, store: IdempotencyStore):
        self._repo = repo
        self._store = store

    def already_executed(self, proposal: ActionProposal, request_id: str | None = None) -> bool:
        return proposal.intent in SIDE_EFFECTS and self._store.get(idempotency_key(proposal, request_id)) is not None

    def execute(self, proposal: ActionProposal, request_id: str | None = None) -> ExecutionResult:
        if proposal.intent not in SIDE_EFFECTS:
            return self._read_only(proposal)

        key = idempotency_key(proposal, request_id)
        previous = self._store.get(key)
        if previous is not None:
            return ExecutionResult("replayed", previous["message"], previous["data"], key)

        if proposal.intent == Intent.ADD_EXAMPLE:
            example = self._repo.append(proposal.params.question, proposal.params.answer)
            result = ExecutionResult("executed", f"added example {example.example_id}",
                                     {"example_id": example.example_id}, key)
        elif proposal.intent == Intent.REMOVE_EXAMPLE:
            example = self._repo.remove(proposal.params.example_id)
            result = ExecutionResult("executed", f"removed example {example.example_id}",
                                     {"example_id": example.example_id, "question": example.question}, key)
        else:  # unreachable while SIDE_EFFECTS and the branches above agree
            raise ValueError(f"no executor for {proposal.intent}")

        self._store.put(key, {"message": result.message, "data": result.data})
        return result

    def _read_only(self, proposal: ActionProposal) -> ExecutionResult:
        if proposal.intent == Intent.DATASET_STATS:
            stats = self._repo.stats()  # computed by code, never taken from the LLM
            return ExecutionResult("answered", f"{stats['examples']} training examples", stats)
        if proposal.intent == Intent.ASK:
            return ExecutionResult("answered", proposal.answer or "")
        raise ValueError(f"no executor for {proposal.intent}")
