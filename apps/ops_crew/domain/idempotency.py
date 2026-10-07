"""Idempotency for side-effecting actions: the same action is executed at most once.

The key is derived from the action itself (intent + normalised params), so a user
re-submitting the same request, or a retry after a crash, replays the stored result
instead of adding a second copy. Callers that deliberately want to repeat an action
(e.g. re-add an example that was removed) pass a new `request_id`, which is part of the key.
"""
import hashlib
import json
from pathlib import Path
from typing import Protocol

from ops_crew.schemas import ActionProposal


def idempotency_key(proposal: ActionProposal, request_id: str | None = None) -> str:
    params = proposal.params.model_dump(exclude_none=True)
    normalised = {k: " ".join(v.split()).lower() if isinstance(v, str) else v for k, v in params.items()}
    payload = json.dumps({"intent": proposal.intent.value, "params": normalised, "request_id": request_id},
                         sort_keys=True)
    return "idem-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


class IdempotencyStore(Protocol):
    def get(self, key: str) -> dict | None: ...
    def put(self, key: str, result: dict) -> None: ...


class InMemoryIdempotencyStore:
    def __init__(self):
        self._results: dict[str, dict] = {}

    def get(self, key):
        return self._results.get(key)

    def put(self, key, result):
        self._results[key] = result


class JsonlIdempotencyStore:
    """Append-only JSONL file; last write for a key wins."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def _load(self) -> dict[str, dict]:
        if not self.path.exists():
            return {}
        entries = (json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line)
        return {e["key"]: e["result"] for e in entries}

    def get(self, key):
        return self._load().get(key)

    def put(self, key, result):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "result": result}, ensure_ascii=False) + "\n")
