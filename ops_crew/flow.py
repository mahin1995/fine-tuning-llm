"""The deterministic pipeline as a CrewAI Flow.

    intake ─► check_input ─┬─ input_invalid ──► reject_input
                           └─ input_ok ──► propose (crew + validation, retries)
                                             ├─ needs_escalation ──► escalate
                                             ├─ low_confidence ────► clarify
                                             └─ proposal_ready ────► authorize_proposal
                                                                       ├─ denied ──────────► reject_proposal
                                                                       ├─ authorized ──────► execute_action
                                                                       └─ approval_required ► request_approval
                                                                                               ├─ approved ─► execute_action
                                                                                               └─ declined ─► decline

Every terminal step writes a "finished" audit record. The flow only orders the steps;
all decisions are made by functions in ops_crew.domain.
Route labels deliberately differ from method names (CrewAI 1.15 rejects a label that
equals a handler name).
"""
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import Any, Callable

from crewai.flow.flow import Flow, listen, or_, router, start
from pydantic import BaseModel, Field, PrivateAttr

from ops_crew.domain.actions import ActionExecutor
from ops_crew.domain.approval import ApprovalGateway, ApprovalRequest
from ops_crew.domain.audit import AuditLog
from ops_crew.domain.inputs import InputRejected, clean_request
from ops_crew.domain.policy import Outcome, authorize
from ops_crew.domain.ports import Proposer, ProposerError
from ops_crew.domain.repository import DatasetRepository
from ops_crew.domain.validation import validate_run
from ops_crew.schemas import ActionProposal, Intent, Role, ToolCallRecord, ValidatedRun
from ops_crew.settings import OpsSettings


@dataclass
class FlowDeps:
    proposer: Proposer
    repo: DatasetRepository
    executor: ActionExecutor
    approvals: ApprovalGateway
    audit: AuditLog
    settings: OpsSettings
    new_correlation_id: Callable[[], str] = field(default=lambda: uuid.uuid4().hex[:12])


class OpsState(BaseModel):
    request: str = ""
    role: Role = Role.VIEWER
    request_id: str | None = None
    correlation_id: str = ""
    clean_text: str = ""
    suspicious: bool = False
    input_error: str | None = None
    attempts: int = 0
    errors: list[str] = Field(default_factory=list)
    proposal: ActionProposal | None = None
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    decision_reasons: list[str] = Field(default_factory=list)
    outcome: str = ""
    message: str = ""
    data: dict = Field(default_factory=dict)


def run_with_timeout(fn: Callable[[], Any], timeout: float):
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(fn)
    try:
        return future.result(timeout=timeout)
    except FuturesTimeout:
        raise ProposerError(f"crew timed out after {timeout:g}s") from None
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


class OpsFlow(Flow[OpsState]):
    _deps: FlowDeps = PrivateAttr()

    def __init__(self, deps: FlowDeps, **kwargs):
        super().__init__(**kwargs)
        self._deps = deps

    # ------------------------------------------------------------------ helpers
    def _audit(self, step: str, **fields):
        self._deps.audit.record(self.state.correlation_id, step, **fields)

    def _finish(self, outcome: str, message: str, **data):
        self.state.outcome = outcome
        self.state.message = message
        self.state.data = data
        self._audit("finished", outcome=outcome, message=message, attempts=self.state.attempts)
        return outcome

    # -------------------------------------------------------------------- steps
    @start()
    def intake(self):
        if not self.state.correlation_id:
            self.state.correlation_id = self._deps.new_correlation_id()
        try:
            clean = clean_request(self.state.request, self._deps.settings.max_request_chars)
        except InputRejected as e:
            self.state.input_error = str(e)
            self._audit("intake", role=self.state.role.value, accepted=False, error=str(e))
            return
        self.state.clean_text = clean.text
        self.state.suspicious = clean.suspicious
        self._audit("intake", role=self.state.role.value, accepted=True, chars=len(clean.text),
                    suspicious=clean.suspicious, injection_markers=list(clean.markers))

    @router(intake)
    def check_input(self):
        return "input_invalid" if self.state.input_error else "input_ok"

    @listen("input_invalid")
    def reject_input(self):
        return self._finish("rejected", f"invalid request: {self.state.input_error}")

    @router("input_ok")
    def propose(self):
        settings = self._deps.settings
        for attempt in range(1, settings.max_retries + 2):
            self.state.attempts = attempt
            try:
                run = run_with_timeout(
                    lambda: self._deps.proposer.propose(self.state.clean_text, self.state.correlation_id),
                    settings.crew_timeout_seconds,
                )
                validated = validate_run(run)
            except Exception as e:  # ProposerError, OutputValidationError, SDK errors: never crash the pipeline
                self._attempt_failed(attempt, type(e).__name__, str(e))
                continue
            return self._accept(validated)
        return "needs_escalation"

    def _attempt_failed(self, attempt, kind, detail):
        self.state.errors.append(f"attempt {attempt}: {kind}: {detail}")
        self._audit("crew_attempt_failed", attempt=attempt, error_type=kind, error=detail[:500])

    def _accept(self, validated: ValidatedRun):
        proposal = validated.proposal
        self.state.proposal = proposal
        self.state.tool_calls = validated.tool_calls
        confidence = min(proposal.confidence, validated.classification.confidence)
        self._audit("proposal_validated", intent=proposal.intent.value, confidence=confidence,
                    issues=proposal.issues, tool_calls=[c.model_dump() for c in validated.tool_calls])
        if confidence < self._deps.settings.confidence_threshold:
            question = proposal.clarifying_question or validated.classification.clarifying_question
            self.state.message = question or ""
            return "low_confidence" if question else "needs_escalation"
        if proposal.intent == Intent.UNKNOWN and proposal.clarifying_question:
            self.state.message = proposal.clarifying_question
            return "low_confidence"
        return "proposal_ready"

    @listen("low_confidence")
    def clarify(self):
        return self._finish("clarification_needed", self.state.message)

    @listen("needs_escalation")
    def escalate(self):
        reason = "; ".join(self.state.errors) or "confidence below threshold and no clarifying question"
        return self._finish("escalated", f"escalated to a human: {reason}")

    @router("proposal_ready")
    def authorize_proposal(self):
        proposal = self.state.proposal
        decision = authorize(proposal, self.state.role, self._deps.repo,
                             suspicious_input=self.state.suspicious,
                             already_executed=self._deps.executor.already_executed(proposal, self.state.request_id))
        self.state.decision_reasons = decision.reasons
        self._audit("authorization", decision=decision.outcome.value, reasons=decision.reasons)
        return {Outcome.ALLOW: "authorized", Outcome.DENY: "denied",
                Outcome.NEEDS_APPROVAL: "approval_required"}[decision.outcome]

    @listen("denied")
    def reject_proposal(self):
        return self._finish("rejected", "; ".join(self.state.decision_reasons))

    @router("approval_required")
    def request_approval(self):
        proposal = self.state.proposal
        params = proposal.params.model_dump(exclude_none=True)
        approved = self._deps.approvals.request(ApprovalRequest(
            correlation_id=self.state.correlation_id,
            intent=proposal.intent.value,
            summary=f"{proposal.intent.value} {params}",
            reasons=tuple(self.state.decision_reasons),
        ))
        self._audit("approval", approved=approved)
        return "approved" if approved else "declined"

    @listen("declined")
    def decline(self):
        return self._finish("declined", "a human declined the action")

    @listen(or_("authorized", "approved"))
    def execute_action(self):
        try:
            result = self._deps.executor.execute(self.state.proposal, self.state.request_id)
        except Exception as e:  # e.g. the example vanished between authorization and execution
            self._audit("execution_failed", error_type=type(e).__name__, error=str(e)[:500])
            return self._finish("failed", f"execution failed: {type(e).__name__}: {e}")
        self._audit("executed", status=result.status, idempotency_key=result.idempotency_key, data=result.data)
        return self._finish("completed", result.message, status=result.status, **result.data)


@dataclass(frozen=True)
class OpsResult:
    correlation_id: str
    outcome: str
    message: str
    data: dict
    attempts: int
    errors: list[str]
    intent: str | None
    params: dict  # the final proposal's parameters (empty if there was no valid proposal)
    tool_calls: list[ToolCallRecord]


def run_request(deps: FlowDeps, request: str, role: Role, request_id: str | None = None) -> OpsResult:
    flow = OpsFlow(deps, suppress_flow_events=True)
    flow.kickoff(inputs={"request": request, "role": role.value, "request_id": request_id})
    s = flow.state
    return OpsResult(
        correlation_id=s.correlation_id, outcome=s.outcome, message=s.message, data=dict(s.data),
        attempts=s.attempts, errors=list(s.errors), intent=s.proposal.intent.value if s.proposal else None,
        params=s.proposal.params.model_dump(exclude_none=True) if s.proposal else {},
        tool_calls=list(s.tool_calls),
    )
