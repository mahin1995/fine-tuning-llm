"""Run golden cases through the full flow and score them.

    python -m ops_crew.evals                         # all cases, profile from env / agents.yaml
    OPS_LLM_PROFILE=openai python -m ops_crew.evals --report outputs/ops/eval_report.json
    python -m ops_crew.evals --case add_duplicate --threshold 1.0

Each case gets a fresh temporary copy of the datasets, so evals never modify data/.
Planned cases (plan, step_outcomes, plan_depends_on) need planning on (OPS_PLANNING, default);
with it off they fail, which shows what planning adds.
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
    reflection_accepted: bool | None = None  # needs the refiner (OPS_REFINE=true)
    plan: list[str] | None = None                    # intents the planner must produce, in order
    plan_depends_on: dict[str, list[str]] = Field(default_factory=dict)  # step id -> expected depends_on
    step_outcomes: list[list[str]] | None = None     # allowed outcomes per executed step ([] = none ran)
    plan_approval_requested: bool | None = None      # whole-plan approval asked (or not)


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
    """Answers every approval request the same way; records what was asked ("plan" or an intent)."""

    def __init__(self, answer: bool):
        self.answer = answer
        self.intents: list[str] = []

    @property
    def requested(self) -> bool:
        return bool(self.intents)

    def request(self, req):
        self.intents.append(req.intent)
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
    repairs: int = 0
    reflection: dict | None = None
    plan: list[str] | None = None          # planned intents (None: not planned)
    plan_repairs: int = 0
    step_outcomes: list[str] = field(default_factory=list)


def _digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _check_plan(exp: Expectation, result: OpsResult, approvals: list[str]) -> list[str]:
    failures = []
    steps = result.plan["steps"] if result.plan else None
    if (exp.plan is not None or exp.plan_depends_on) and steps is None:
        failures.append("no plan (planning disabled, or the planner failed)")
    elif steps is not None:
        intents = [s["intent"] for s in steps]
        if exp.plan is not None and intents != exp.plan:
            failures.append(f"plan intents {intents} != expected {exp.plan}")
        depends = {s["id"]: s["depends_on"] for s in steps}
        for step_id, expected in exp.plan_depends_on.items():
            if depends.get(step_id) != expected:
                failures.append(f"{step_id} depends_on {depends.get(step_id)} != expected {expected}")
    if exp.step_outcomes is not None:
        got = [s.outcome for s in result.steps]
        if len(got) != len(exp.step_outcomes) or any(o not in allowed for o, allowed in zip(got, exp.step_outcomes)):
            failures.append(f"step outcomes {got} do not match {exp.step_outcomes}")
    if exp.plan_approval_requested is not None and ("plan" in approvals) != exp.plan_approval_requested:
        failures.append(f"plan approval requested={'plan' in approvals}, expected {exp.plan_approval_requested}")
    return failures


def check(case: GoldenCase, result: OpsResult, approvals: list[str], dataset_changed: bool) -> list[str]:
    """`approvals`: what each approval request was for, in order ("plan" or an intent)."""
    exp, failures = case.expect, []
    approval_requested = bool(approvals)
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
    if exp.reflection_accepted is not None:
        if result.reflection is None:
            failures.append("no reflection ran (is the refiner enabled?)")
        elif result.reflection["accepted"] != exp.reflection_accepted:
            failures.append(f"reflection accepted={result.reflection['accepted']}, expected {exp.reflection_accepted}")
    return failures + _check_plan(exp, result, approvals)


def run_case(case: GoldenCase, make_deps: Callable[[GoldenCase, ScriptedApproval], FlowDeps]) -> CaseResult:
    approval = ScriptedApproval(case.approve)
    deps = make_deps(case, approval)
    before = _digest(deps.repo.train_path)
    result = run_request(deps, case.request, case.role)
    changed = _digest(deps.repo.train_path) != before
    failures = check(case, result, approval.intents, changed)
    plan = [s["intent"] for s in result.plan["steps"]] if result.plan else None
    return CaseResult(case.id, not failures, failures, result.outcome, result.intent,
                      [c.tool for c in result.tool_calls], result.message, result.errors,
                      result.repairs, result.reflection, plan,
                      result.plan["repairs"] if result.plan else 0, [s.outcome for s in result.steps])


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

    @property
    def refinement_stats(self) -> dict:
        reflected = [r.reflection for r in self.results if r.reflection]
        scores = [r["score"] for r in reflected if r["score"] is not None]
        return {
            "cases_self_corrected": sum(r.repairs > 0 for r in self.results),
            "cases_reflected": len(reflected),
            "reflection_accepted": sum(r["accepted"] for r in reflected),
            "reflection_revised": sum(r["rounds"] > 0 for r in reflected),
            "mean_reflection_score": round(sum(scores) / len(scores), 3) if scores else None,
        }

    @property
    def planning_stats(self) -> dict:
        planned = [r for r in self.results if r.plan is not None]
        return {
            "cases_planned": len(planned),
            "multi_step_plans": sum(len(r.plan) > 1 for r in planned),
            "plans_repaired": sum(r.plan_repairs > 0 for r in planned),
        }

    def to_dict(self):
        return {"pass_rate": self.pass_rate, "threshold": self.threshold, "passed": self.passed,
                "refinement": self.refinement_stats, "planning": self.planning_stats,
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
        detail = f"plan={r.plan} steps={r.step_outcomes}" if r.plan and len(r.plan) > 1 else f"intent={r.intent}"
        lines.append(f"{mark}  {r.case_id:<28} outcome={r.outcome:<21} {detail} tools={r.tools}")
        lines += [f"        - {f}" for f in r.failures]
    stats = report.refinement_stats
    lines.append(f"\nself-corrected: {stats['cases_self_corrected']} case(s); reflected: {stats['cases_reflected']} "
                 f"(accepted {stats['reflection_accepted']}, revised {stats['reflection_revised']}, "
                 f"mean score {stats['mean_reflection_score']})")
    planning = report.planning_stats
    lines.append(f"planned: {planning['cases_planned']} case(s); multi-step: {planning['multi_step_plans']}; "
                 f"plans repaired: {planning['plans_repaired']}")
    lines.append(f"pass rate {report.pass_rate:.0%} (threshold {report.threshold:.0%}): "
                 f"{'OK' if report.passed else 'BELOW THRESHOLD'}")
    return "\n".join(lines)


def write_report(report: SuiteReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
