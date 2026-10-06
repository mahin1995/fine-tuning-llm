"""Run golden cases through the full flow and score them.

    python -m ops_crew.evals                         # all cases, profile from env / agents.yaml
    OPS_LLM_PROFILE=openai python -m ops_crew.evals --report outputs/ops/eval_report.json
    python -m ops_crew.evals --case add_duplicate --threshold 1.0

Each case gets a fresh temporary copy of the datasets, so evals never modify data/.
Exit code 0 when the pass rate reaches the threshold, 1 otherwise.
"""
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ops_crew.flow import FlowDeps, OpsResult, run_request
from ops_crew.schemas import Role

GOLDEN_PATH = Path(__file__).parent / "golden.yaml"


class Expectation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    intent: str | None = None
    outcomes: list[str]
    params_contains: dict[str, str] = Field(default_factory=dict)
    required_tools: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    approval_requested: bool | None = None
    dataset_unchanged: bool | None = None


class GoldenCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    role: Role
    request: str
    approve: bool = False
    expect: Expectation


class GoldenSuite(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pass_threshold: float = Field(ge=0, le=1)
    cases: list[GoldenCase]


def load_suite(path: Path = GOLDEN_PATH) -> GoldenSuite:
    suite = GoldenSuite.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
    ids = [c.id for c in suite.cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case ids")
    return suite


class ScriptedApproval:
    def __init__(self, answer: bool):
        self.answer = answer
        self.requested = False

    def request(self, req):
        self.requested = True
        return self.answer


@dataclass
class CaseResult:
    case_id: str
    passed: bool
    failures: list[str]
    outcome: str
    intent: str | None
    tools: list[str]
    message: str
    errors: list[str] = field(default_factory=list)


def _digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(case: GoldenCase, result: OpsResult, approval_requested: bool, dataset_changed: bool) -> list[str]:
    exp, failures = case.expect, []
    if exp.intent is not None and result.intent != exp.intent:
        failures.append(f"intent {result.intent!r} != expected {exp.intent!r}")
    if result.outcome not in exp.outcomes:
        failures.append(f"outcome {result.outcome!r} not in {exp.outcomes}")
    for key, needle in exp.params_contains.items():
        value = result.params.get(key) or ""
        if needle.lower() not in value.lower():
            failures.append(f"param {key!r} does not contain {needle!r}")
    used = {c.tool for c in result.tool_calls}
    for tool in exp.required_tools:
        if tool not in used:
            failures.append(f"required tool {tool!r} was not called")
    for tool in exp.forbidden_tools:
        if tool in used:
            failures.append(f"forbidden tool {tool!r} was called")
    if exp.approval_requested is not None and approval_requested != exp.approval_requested:
        failures.append(f"approval requested={approval_requested}, expected {exp.approval_requested}")
    if exp.dataset_unchanged is not None and dataset_changed == exp.dataset_unchanged:
        failures.append("dataset changed" if dataset_changed else "dataset was expected to change but did not")
    return failures


def run_case(case: GoldenCase, make_deps: Callable[[GoldenCase, ScriptedApproval], FlowDeps]) -> CaseResult:
    approval = ScriptedApproval(case.approve)
    deps = make_deps(case, approval)
    before = _digest(deps.repo.train_path)
    result = run_request(deps, case.request, case.role)
    changed = _digest(deps.repo.train_path) != before
    failures = check(case, result, approval.requested, changed)
    return CaseResult(case.id, not failures, failures, result.outcome, result.intent,
                      [c.tool for c in result.tool_calls], result.message, result.errors)


@dataclass
class SuiteReport:
    results: list[CaseResult]
    threshold: float

    @property
    def pass_rate(self) -> float:
        return sum(r.passed for r in self.results) / len(self.results) if self.results else 0.0

    @property
    def passed(self) -> bool:
        return self.pass_rate >= self.threshold

    def to_dict(self):
        return {"pass_rate": self.pass_rate, "threshold": self.threshold, "passed": self.passed,
                "cases": [asdict(r) for r in self.results]}


def run_suite(suite: GoldenSuite, make_deps, threshold: float | None = None, only: list[str] | None = None,
              on_result: Callable[[CaseResult], None] | None = None) -> SuiteReport:
    cases = [c for c in suite.cases if not only or c.id in only]
    results = []
    for case in cases:
        result = run_case(case, make_deps)
        results.append(result)
        if on_result:
            on_result(result)
    return SuiteReport(results, suite.pass_threshold if threshold is None else threshold)


def format_report(report: SuiteReport) -> str:
    lines = []
    for r in report.results:
        mark = "PASS" if r.passed else "FAIL"
        lines.append(f"{mark}  {r.case_id:<28} outcome={r.outcome:<21} intent={r.intent} tools={r.tools}")
        lines += [f"        - {f}" for f in r.failures]
    lines.append(f"\npass rate {report.pass_rate:.0%} (threshold {report.threshold:.0%}): "
                 f"{'OK' if report.passed else 'BELOW THRESHOLD'}")
    return "\n".join(lines)


def write_report(report: SuiteReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
