"""Memory consolidation: turn many past requests (episodes) into a few lessons for a human.

    python -m ops_crew.consolidate                     # last 30 days -> outputs/ops/consolidation/
    python -m ops_crew.consolidate --days 7 --prune    # and forget episodes past their retention

Deterministic, no LLM, and nothing is applied automatically. It writes:

    report.md                  outcomes per intent, the most common failure messages, groups of
                               similar requests that did not complete
    golden_candidates.yaml     one eval case per group of failed requests, with what was observed:
                               set the behaviour you WANT, then copy it into evals/golden.yaml
    training_candidates.jsonl  questions whose answer passed reflection with a high score, in the
                               dataset format: review them, then add the good ones with add_example

Episodic memory (raw, short-lived) becomes knowledge (eval cases, training data) only through a
human. --prune then forgets the raw episodes, so memory stays bounded.
"""
import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import yaml

from ops_crew.domain.ports import RequestRecord

UNSUCCESSFUL = ("rejected", "declined", "escalated", "clarification_needed", "failed", "partially_completed")
MIN_TRAINING_SCORE = 0.9


@dataclass
class Consolidation:
    report: str
    golden_candidates: dict                # {"cases": [...], "observed": {...}}
    training_candidates: list[dict]        # dataset rows: {"messages": [...]}


def _short(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def consolidate(records: list[RequestRecord], cluster: Callable[[list[str]], list[list[int]]], *,
                days: int, now: datetime, min_training_score: float = MIN_TRAINING_SCORE) -> Consolidation:
    """`records` newest first; `cluster` groups similar texts (returns lists of indices)."""
    failed = [r for r in records if r.outcome in UNSUCCESSFUL]
    groups = sorted(cluster([r.request for r in failed]) if failed else [], key=len, reverse=True)
    training = [r for r in records if r.intent == "ask" and r.outcome == "completed" and not r.reply_to
                and r.reflection_score is not None and r.reflection_score >= min_training_score]

    lines = [f"# Consolidation report ({now:%Y-%m-%d}, last {days} days)", "",
             f"{len(records)} request(s), {len(failed)} not completed, {len(groups)} group(s) of similar failures, "
             f"{len(training)} training candidate(s).", "", "## Outcomes", "", "| outcome | requests |", "|---|---|"]
    lines += [f"| {o} | {n} |" for o, n in Counter(r.outcome for r in records).most_common()]
    lines += ["", "## Intent x outcome", "", "| intent | outcome | requests |", "|---|---|---|"]
    by_intent = Counter((r.intent, r.outcome) for r in records)
    lines += [f"| {i or '-'} | {o} | {n} |" for (i, o), n in by_intent.most_common()]
    lines += ["", "## Most common failure messages", ""]
    messages = Counter(_short(r.message, 120) for r in failed)
    lines += [f"- {n} x `{m}`" for m, n in messages.most_common(10)] or ["- none"]
    lines += ["", "## Groups of similar failed requests", ""]

    cases, observed = [], {}
    for n, group in enumerate(groups, start=1):
        members = [failed[i] for i in group]
        first = members[0]  # newest of the group
        outcomes = Counter(m.outcome for m in members)
        lines.append(f"{n}. {len(members)} x {dict(outcomes)}: `{_short(first.request, 160)}`")
        case_id = f"candidate_{n}"
        cases.append({"id": case_id, "role": first.role, "request": first.request, "approve": False,
                      "expect": {"outcomes": [first.outcome], "dataset_unchanged": True}})
        observed[case_id] = {"outcomes": dict(outcomes), "message": _short(first.message, 200),
                             "correlation_ids": [m.correlation_id for m in members[:5]]}
    if not groups:
        lines.append("- none")
    lines += ["", "## Training candidates (review before adding)", ""]
    lines += [f"- {r.correlation_id} (reflection score {r.reflection_score:.2f}): `{_short(r.request, 120)}`"
              for r in training] or ["- none"]

    rows = [{"messages": [{"role": "user", "content": r.request}, {"role": "assistant", "content": r.message}]}
            for r in training]
    return Consolidation("\n".join(lines) + "\n", {"cases": cases, "observed": observed}, rows)


GOLDEN_HEADER = """\
# Candidate eval cases from failed requests (python -m ops_crew.consolidate).
# REVIEW EACH ONE: `expect` holds what was OBSERVED, not necessarily what is right. Set the behaviour
# you want (outcomes, approval_requested, dataset_unchanged, ...), then copy the case into
# apps/ops_crew/evals/golden.yaml. `observed` is for reference and is not part of a case.
"""


def write(result: Consolidation, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    report, golden, training = out_dir / "report.md", out_dir / "golden_candidates.yaml", \
        out_dir / "training_candidates.jsonl"
    report.write_text(result.report, encoding="utf-8")
    golden.write_text(GOLDEN_HEADER + yaml.safe_dump(result.golden_candidates, sort_keys=False, allow_unicode=True,
                                                     width=100), encoding="utf-8")
    training.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in result.training_candidates),
                        encoding="utf-8")
    return [report, golden, training]


def main(argv=None) -> int:
    from ops_crew.memory_adapters import SqliteEpisodeMemory, cluster_requests, load_memory_config
    from ops_crew.settings import OpsSettings

    p = argparse.ArgumentParser(prog="python -m ops_crew.consolidate", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--days", type=int, default=30, help="look at requests from the last N days")
    p.add_argument("--out", type=Path, default=None, help="output directory (default: <state_dir>/consolidation)")
    p.add_argument("--prune", action="store_true", help="afterwards, forget episodes past their retention")
    args = p.parse_args(argv)

    settings = OpsSettings()
    if not settings.memory_db_path.exists():
        print(f"no episodes yet ({settings.memory_db_path} does not exist)")
        return 0
    config = load_memory_config(settings.config_dir)
    memory = SqliteEpisodeMemory(settings.memory_db_path)
    now = datetime.now(timezone.utc)
    records = memory.requests(since=now - timedelta(days=args.days))
    result = consolidate(records, lambda texts: cluster_requests(texts, config), days=args.days, now=now)
    for path in write(result, args.out or settings.state_dir / "consolidation"):
        print(f"wrote {path}")
    if args.prune:
        deleted = memory.prune(now, config.episodes.retention_days)
        print("pruned: " + ", ".join(f"{n} {kind} episode(s)" for kind, n in deleted.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
