"""What the deterministic flow needs from the probabilistic layer."""
from typing import Protocol

from ops_crew.schemas import CrewRun


class ProposerError(RuntimeError):
    """The crew could not produce output: provider failure, timeout, conversion failure, ..."""


class Proposer(Protocol):
    def propose(self, request: str, correlation_id: str) -> CrewRun:
        """Run the crew on one request. Must not have side effects. Raises ProposerError."""
        ...
