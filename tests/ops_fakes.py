"""Shared fakes for ops_crew tests: scripted proposer, crew-run builder, dataset copies."""
import json
import shutil

from ops_crew.domain.actions import ActionExecutor
from ops_crew.domain.audit import AuditLog, MemorySink
from ops_crew.domain.idempotency import InMemoryIdempotencyStore
from ops_crew.domain.ports import ProposerError
from ops_crew.domain.repository import DatasetRepository
from ops_crew.schemas import CrewRun, ToolCallRecord
from ops_crew.settings import OpsSettings

from conftest import EVAL_DATA, TRAIN_DATA


def make_run(intent="ask", *, params=None, answer=None, confidence=0.9, classifier_confidence=None,
             issues=None, clarifying_question=None, tool_calls=None, **overrides) -> CrewRun:
    """A well-formed crew run; override any task output with raw text via overrides."""
    params = params or {}
    outputs = {
        "classify_request": json.dumps({
            "intent": intent, "params": params,
            "confidence": confidence if classifier_confidence is None else classifier_confidence,
            "clarifying_question": clarifying_question,
        }),
        "research_context": json.dumps({"summary": "context gathered", "related_example_ids": []}),
        "review_proposal": json.dumps({
            "intent": intent, "params": params, "answer": answer, "rationale": "reviewed",
            "issues": issues or [], "confidence": confidence, "clarifying_question": clarifying_question,
        }),
    }
    outputs.update(overrides)
    return CrewRun(outputs=outputs, tool_calls=[ToolCallRecord(**c) for c in (tool_calls or [])])


class FakeProposer:
    """Returns (or raises) scripted results in order; records every call."""

    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def propose(self, request, correlation_id):
        self.calls.append((request, correlation_id))
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        if callable(result):
            return result()
        return result


class RecordingApprovals:
    def __init__(self, answer: bool):
        self.answer = answer
        self.requests = []

    def request(self, req):
        self.requests.append(req)
        return self.answer


def copy_datasets(tmp_path):
    train, evals = tmp_path / "train.jsonl", tmp_path / "eval.jsonl"
    shutil.copy(TRAIN_DATA, train)
    shutil.copy(EVAL_DATA, evals)
    return DatasetRepository(train, evals)


def build_deps(tmp_path, proposer, approvals=None, **settings_overrides):
    from ops_crew.flow import FlowDeps  # needs crewai

    repo = copy_datasets(tmp_path)
    sink = MemorySink()
    ids = iter(f"cid{i:03d}" for i in range(1000))
    deps = FlowDeps(
        proposer=proposer,
        repo=repo,
        executor=ActionExecutor(repo, InMemoryIdempotencyStore()),
        approvals=approvals or RecordingApprovals(False),
        audit=AuditLog([sink]),
        settings=OpsSettings(train_data=repo.train_path, eval_data=repo.eval_path, state_dir=tmp_path,
                             **settings_overrides),
        new_correlation_id=lambda: next(ids),
    )
    return deps, sink


__all__ = ["make_run", "FakeProposer", "RecordingApprovals", "copy_datasets", "build_deps", "ProposerError"]
