"""Shared fakes for ops_crew tests: scripted planner / proposer / refiner, builders, dataset copies."""
import json
import shutil
from datetime import datetime, timezone

from ops_crew.domain.actions import ActionExecutor
from ops_crew.domain.audit import AuditLog, MemorySink
from ops_crew.domain.idempotency import InMemoryIdempotencyStore
from ops_crew.domain.ports import ProposerError
from ops_crew.domain.repository import DatasetRepository
from ops_crew.schemas import CrewRun, Plan, ToolCallRecord
from ops_crew.settings import OpsSettings

from conftest import EVAL_DATA, TRAIN_DATA


def make_run(intent="ask", *, params=None, answer=None, confidence=0.9, classifier_confidence=None,
             issues=None, clarifying_question=None, tool_calls=None, related_ids=None, **overrides) -> CrewRun:
    """A well-formed crew run; override any task output with raw text via overrides."""
    params = params or {}
    outputs = {
        "classify_request": json.dumps({
            "intent": intent, "params": params,
            "confidence": confidence if classifier_confidence is None else classifier_confidence,
            "clarifying_question": clarifying_question,
        }),
        "research_context": json.dumps({"summary": "context gathered", "related_example_ids": related_ids or []}),
        "review_proposal": json.dumps({
            "intent": intent, "params": params, "answer": answer, "rationale": "reviewed",
            "issues": issues or [], "confidence": confidence, "clarifying_question": clarifying_question,
        }),
    }
    outputs.update(overrides)
    calls = [ToolCallRecord(**c) for c in (tool_calls or [])]
    if related_ids:  # the ids the research tools returned (the shared working state the checks trust)
        facts = {"example_ids": list(related_ids)}
        search = next((i for i, c in enumerate(calls) if c.tool == "search_training_data" and not c.facts), None)
        if search is None:
            calls.append(ToolCallRecord(agent="researcher", tool="search_training_data", arguments={}, facts=facts))
        else:
            calls[search] = calls[search].model_copy(update={"facts": facts})
    return CrewRun(outputs=outputs, tool_calls=calls)


def make_plan(*steps, rationale="split by intent") -> dict:
    """A plan as the planner would return it: steps are (intent, excerpt, *depends_on)."""
    return {"steps": [{"id": f"s{i}", "intent": intent, "excerpt": excerpt, "depends_on": list(deps)}
                      for i, (intent, excerpt, *deps) in enumerate(steps, start=1)],
            "rationale": rationale}


class FakePlanner:
    """Returns (or raises) scripted plans in order: dicts (JSON-encoded), raw strings or exceptions."""

    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def plan(self, request, correlation_id):
        self.calls.append((request, correlation_id))
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result if isinstance(result, str) else json.dumps(result)


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


class FakeRefiner:
    """Scripted Refiner: `repairs` (proposals or exceptions) and `reviews` (ReflectionOutcome or exceptions)."""

    def __init__(self, repairs=(), reviews=(), plans=()):
        self.repairs = list(repairs)
        self.reviews = list(reviews)
        self.plans = list(plans)
        self.repair_calls = []
        self.review_calls = []
        self.plan_repair_calls = []

    def _next(self, queue):
        item = queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def repair(self, req):
        self.repair_calls.append(req)
        return self._next(self.repairs)

    def review_answer(self, request, proposal, findings):
        self.review_calls.append(("ask", proposal))
        return self._next(self.reviews)

    def assess_example(self, request, proposal, findings):
        self.review_calls.append(("add_example", proposal))
        return self._next(self.reviews)

    def repair_plan(self, req):
        self.plan_repair_calls.append(req)
        item = self._next(self.plans)
        return Plan.model_validate(item) if isinstance(item, dict) else item


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


T0 = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)


def build_deps(tmp_path, proposer, approvals=None, refiner=None, planner=None, memory=False, **settings_overrides):
    """`memory=True` wires the real DatasetSimilarity (hashing embedder: offline, deterministic) and
    SqliteEpisodeMemory in tmp_path, with the clock fixed at T0."""
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
        refiner=refiner,
        planner=planner,
    )
    if memory:
        from ops_crew.memory_adapters import DatasetSimilarity, SqliteEpisodeMemory, load_memory_config

        deps.similarity = DatasetSimilarity.from_config(repo, load_memory_config(deps.settings.config_dir, env={}),
                                                        env={})
        deps.episodes = SqliteEpisodeMemory(deps.settings.memory_db_path)
        deps.now = lambda: T0  # tests move time by replacing deps.now
    return deps, sink


__all__ = ["make_run", "make_plan", "FakePlanner", "FakeProposer", "FakeRefiner", "RecordingApprovals",
           "copy_datasets", "build_deps", "ProposerError", "T0"]
