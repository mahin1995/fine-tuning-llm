"""Dataset Ops Assistant CLI (composition root: the only place real objects are wired).

    python -m ops_crew "How many examples are in the training data?"
    python -m ops_crew --role editor "Add Q: What is a Spring profile? A: A named set of beans ..."
    python -m ops_crew --role admin "Remove the example about bean scopes"     # asks for approval
    python -m ops_crew --role admin "Remove the N+1 example and add: Q: ... A: ..."  # planned: 2 steps
    python -m ops_crew --role admin --reply-to 3f2a9c1b7d4e "the one about bean scopes"  # answer a question
    OPS_LLM_PROFILE=openai python -m ops_crew --json "What does REQUIRES_NEW do?"

The role comes from the caller (here: the CLI flag; in a service: authentication), never
from the request text. Approval prompts are shown only on an interactive terminal; in
non-interactive runs every approval is denied.

A planner splits requests that ask for several actions into steps (OPS_PLANNING=false turns it
off); a plan that changes data is shown for approval as a whole before its first step runs.

Exit code: 0 for completed / clarification_needed, 1 otherwise (including partially_completed).
"""
import os

# Before CrewAI is imported anywhere: no telemetry / tracing unless explicitly enabled.
for _var, _value in (("CREWAI_DISABLE_TELEMETRY", "true"), ("OTEL_SDK_DISABLED", "true"),
                     ("CREWAI_TRACING_ENABLED", "false")):
    os.environ.setdefault(_var, _value)

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from dataclasses import asdict  # noqa: E402

from ops_crew.schemas import Role  # noqa: E402
from ops_crew.settings import OpsSettings  # noqa: E402

OK_OUTCOMES = {"completed", "clarification_needed"}


def build_deps(settings: OpsSettings, *, interactive: bool, log_to_stderr: bool, verbose: bool):
    from ops_crew.crew.config import load_config
    from ops_crew.crew.llm import ProfileLLMFactory
    from ops_crew.crew.planner import CrewPlanner
    from ops_crew.crew.proposer import CrewProposer
    from ops_crew.crew.tools import ReadOnlyDataset
    from ops_crew.domain.actions import ActionExecutor
    from ops_crew.domain.approval import ConsoleApproval, DenyAllApprovals
    from ops_crew.domain.audit import AuditLog, JsonlSink, StreamSink
    from ops_crew.domain.idempotency import JsonlIdempotencyStore
    from ops_crew.domain.repository import DatasetRepository
    from ops_crew.flow import FlowDeps

    sinks = [JsonlSink(settings.audit_log_path)] + ([StreamSink()] if log_to_stderr else [])
    audit = AuditLog(sinks)
    config = load_config(settings.config_dir)
    env = dict(os.environ)
    if settings.llm_profile:
        env.setdefault("OPS_LLM_PROFILE", settings.llm_profile)
    llms = ProfileLLMFactory(config, env, on_fallback=lambda model, err: audit.record(
        "-", "llm_fallback", model=model, error=f"{type(err).__name__}: {err}"[:300]))
    llms.check()
    repo = DatasetRepository(settings.train_data, settings.eval_data)
    similarity = episodes = None
    if settings.memory:
        from ops_crew.memory_adapters import DatasetSimilarity, SqliteEpisodeMemory, load_memory_config

        similarity = DatasetSimilarity.from_config(repo, load_memory_config(settings.config_dir, env), env)
        episodes = SqliteEpisodeMemory(settings.memory_db_path)
    refiner = None
    if settings.refine:
        from ops_crew.refinement import PydanticAIRefiner

        refiner = PydanticAIRefiner.from_config(settings.config_dir, config, env)
    return FlowDeps(
        proposer=CrewProposer(config, llms, ReadOnlyDataset(repo, similarity), verbose=verbose),
        repo=repo,
        executor=ActionExecutor(repo, JsonlIdempotencyStore(settings.idempotency_path)),
        approvals=ConsoleApproval() if interactive else DenyAllApprovals(),
        audit=audit,
        settings=settings,
        refiner=refiner,
        planner=CrewPlanner(config, llms, verbose=verbose) if settings.planning else None,
        similarity=similarity,
        episodes=episodes,
    )


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m ops_crew", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("request", help="what you want, in plain language")
    p.add_argument("--role", choices=[r.value for r in Role], default=Role.VIEWER.value)
    p.add_argument("--request-id", default=None, help="idempotency scope; reuse to make a retry a no-op")
    p.add_argument("--reply-to", default=None, metavar="CORRELATION_ID",
                   help="answer the clarifying question an earlier request asked (episodic memory)")
    p.add_argument("--json", action="store_true", help="print the result as JSON")
    p.add_argument("--log", action="store_true", help="also print audit records to stderr")
    p.add_argument("--verbose", action="store_true", help="print CrewAI agent output")
    p.add_argument("--no-input", action="store_true", help="never prompt; deny all approvals")
    args = p.parse_args(argv)

    from ops_crew.crew.config import ConfigError
    from ops_crew.flow import run_request

    settings = OpsSettings()
    try:
        deps = build_deps(settings, interactive=sys.stdin.isatty() and not args.no_input,
                          log_to_stderr=args.log, verbose=args.verbose)
    except ConfigError as e:
        print(f"configuration error: {e}", file=sys.stderr)
        return 2
    result = run_request(deps, args.request, Role(args.role), args.request_id, reply_to=args.reply_to)

    if args.json:
        print(json.dumps(asdict(result), indent=2, ensure_ascii=False, default=str))
    elif len(result.steps) > 1:
        print(f"[{result.outcome}] plan {result.plan['hash']}, {len(result.steps)} steps")
        for step in result.steps:
            print(f"  {step.step_id} {step.intent} [{step.outcome}] {step.message}")
    else:
        print(f"[{result.outcome}] {result.message}")
    if not args.json:
        print(f"correlation id: {result.correlation_id} (audit log: {settings.audit_log_path})", file=sys.stderr)
        if result.outcome == "clarification_needed" and deps.episodes is not None:
            print(f'answer with: python -m ops_crew --role {args.role} --reply-to {result.correlation_id} "..."',
                  file=sys.stderr)
    return 0 if result.outcome in OK_OUTCOMES else 1


if __name__ == "__main__":
    sys.exit(main())
