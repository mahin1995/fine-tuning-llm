"""Human approval for destructive or risky actions."""
from dataclasses import dataclass
from typing import Callable, Protocol


@dataclass(frozen=True)
class ApprovalRequest:
    correlation_id: str
    intent: str
    summary: str
    reasons: tuple[str, ...]


class ApprovalGateway(Protocol):
    def request(self, req: ApprovalRequest) -> bool:
        """Return True only if a human explicitly approved."""
        ...


class DenyAllApprovals:
    """Safe default for non-interactive runs (evals, batch): nothing risky executes."""

    def request(self, req):
        return False


class ConsoleApproval:
    def __init__(self, ask: Callable[[str], str] = input, show: Callable[[str], None] = print):
        self._ask = ask
        self._show = show

    def request(self, req):
        self._show(f"\nAPPROVAL NEEDED [{req.correlation_id}] {req.intent}: {req.summary}")
        for reason in req.reasons:
            self._show(f"  - {reason}")
        try:
            answer = self._ask("approve? type 'yes' to approve: ")
        except EOFError:
            return False
        return answer.strip().lower() == "yes"  # anything else, including 'y', is a no
