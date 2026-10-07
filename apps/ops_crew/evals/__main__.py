"""python -m ops_crew.evals [--threshold 0.8] [--case ID ...] [--report PATH] [--cases PATH]"""
import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

from ops_crew.domain.actions import ActionExecutor
from ops_crew.domain.audit import AuditLog, JsonlSink
from ops_crew.domain.idempotency import InMemoryIdempotencyStore
from ops_crew.domain.repository import DatasetRepository
from ops_crew.evals.runner import GOLDEN_PATH, format_report, load_suite, run_suite, write_report
from ops_crew.settings import OpsSettings


def make_deps_factory(settings: OpsSettings, workdir: Path, verbose: bool):
    from ops_crew.crew.config import load_config
    from ops_crew.crew.llm import ProfileLLMFactory
    from ops_crew.crew.proposer import CrewProposer
    from ops_crew.crew.tools import ReadOnlyDataset
    from ops_crew.flow import FlowDeps

    config = load_config(settings.config_dir)
    llms = ProfileLLMFactory(config, dict(os.environ))
    llms.check()  # fail fast on missing keys instead of 13 escalated cases
    refiner = None
    if settings.refine:
        from ops_crew.refinement import PydanticAIRefiner

        refiner = PydanticAIRefiner.from_config(settings.config_dir, config)

    def make_deps(case, approval):
        case_dir = workdir / case.id
        case_dir.mkdir(parents=True, exist_ok=True)
        train, evals = case_dir / "train.jsonl", case_dir / "eval.jsonl"
        shutil.copy(settings.train_data, train)
        shutil.copy(settings.eval_data, evals)
        repo = DatasetRepository(train, evals)
        return FlowDeps(
            proposer=CrewProposer(config, llms, ReadOnlyDataset(repo), verbose=verbose),
            repo=repo,
            executor=ActionExecutor(repo, InMemoryIdempotencyStore()),
            approvals=approval,
            audit=AuditLog([JsonlSink(workdir / "audit.jsonl")]),
            settings=settings,
            refiner=refiner,
        )

    return make_deps


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m ops_crew.evals", description=__doc__)
    p.add_argument("--cases", type=Path, default=GOLDEN_PATH)
    p.add_argument("--case", action="append", help="run only this case id (repeatable)")
    p.add_argument("--threshold", type=float, default=None, help="override pass_threshold from the cases file")
    p.add_argument("--report", type=Path, default=None, help="write a JSON report here")
    p.add_argument("--verbose", action="store_true", help="print CrewAI agent output")
    args = p.parse_args(argv)

    from ops_crew.crew.config import ConfigError

    suite = load_suite(args.cases)
    settings = OpsSettings()
    with tempfile.TemporaryDirectory(prefix="ops-evals-") as tmp:
        try:
            make_deps = make_deps_factory(settings, Path(tmp), args.verbose)
        except ConfigError as e:
            print(f"configuration error: {e}", file=sys.stderr)
            return 2
        report = run_suite(suite, make_deps, args.threshold, args.case,
                           on_result=lambda r: print(f"{'PASS' if r.passed else 'FAIL'} {r.case_id}", file=sys.stderr))
    print(format_report(report))
    if args.report:
        write_report(report, args.report)
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
