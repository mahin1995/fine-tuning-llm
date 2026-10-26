"""ops_crew episodic memory (requests and declines remembered in SQLite) and consolidation."""
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest
import yaml

pytest.importorskip("pydantic_settings")  # ops_crew deps (apps/requirements.txt)

from ops_crew.consolidate import consolidate, write  # noqa: E402
from ops_crew.domain.followup import followup_text, reply_problem  # noqa: E402
from ops_crew.domain.ports import RequestRecord  # noqa: E402
from ops_crew.domain.repository import example_id_for  # noqa: E402
from ops_crew.memory_adapters import SqliteEpisodeMemory, cluster_requests, load_memory_config  # noqa: E402
from ops_crew.settings import OpsSettings  # noqa: E402
from ops_fakes import T0  # noqa: E402

SCOPES_ID = example_id_for("What bean scopes does Spring support and which is the default?")
NEW_Q = "What is the difference between @Value and @ConfigurationProperties?"
NEW_A = "@ConfigurationProperties binds a whole group of properties to a typed bean; @Value injects one."
TTL = timedelta(minutes=60)


def record(cid, outcome="clarification_needed", minutes=0, **kw):
    defaults = {"role": "admin", "request": "Remove that example.", "message": "Which example do you want to remove?"}
    return RequestRecord(correlation_id=cid, ts=T0 + timedelta(minutes=minutes), outcome=outcome, **{**defaults, **kw})


# ---------------------------------------------------------------- follow-up

def test_reply_rules():
    asked = record("cid000")
    assert reply_problem(asked, "cid000", T0 + timedelta(minutes=59), TTL) is None
    assert reply_problem(None, "nope", T0, TTL) == "nothing to reply to: no request nope in memory"
    assert reply_problem(record("cid1", "completed"), "cid1", T0, TTL) == \
        "request cid1 did not ask a question (its outcome was completed)"
    assert reply_problem(asked, "cid000", T0 + timedelta(minutes=61), TTL) == \
        "the question from request cid000 has expired (older than 60 minutes); send a new request"
    assert followup_text(asked, "the bean scopes one") == (
        "Earlier request: Remove that example.\nClarifying question: Which example do you want to remove?\n"
        "Answer: the bean scopes one")


# ------------------------------------------------------------------- adapter

@pytest.fixture
def episodes(tmp_path):
    return SqliteEpisodeMemory(tmp_path / "memory.db")


def test_requests_round_trip(episodes):
    episodes.record_request(record("cid000", plan_intents=("ask", "add_example"), step_outcomes=("completed",),
                                   reflection_score=0.8, reply_to="cid9"))
    assert episodes.get_request("cid000") == record("cid000", plan_intents=("ask", "add_example"),
                                                    step_outcomes=("completed",), reflection_score=0.8,
                                                    reply_to="cid9")
    assert episodes.get_request("missing") is None
    episodes.record_request(record("cid001", "completed", minutes=5))
    assert [r.correlation_id for r in episodes.requests()] == ["cid001", "cid000"]
    assert [r.correlation_id for r in episodes.requests(since=T0 + timedelta(minutes=1))] == ["cid001"]


def test_declines_and_retention(episodes):
    episodes.record_decline("idem-abc", "cid000", T0)
    episodes.record_decline("idem-abc", "cid004", T0 + timedelta(days=2))
    assert episodes.declined_before("idem-abc") == T0 + timedelta(days=2)
    assert episodes.declined_before("idem-other") is None
    assert episodes.get_request("cid000:idem-abc") is None  # a decline is not a request
    episodes.record_request(record("old", "completed", minutes=-60 * 24 * 40))
    episodes.record_request(record("new", "completed"))
    assert episodes.prune(T0 + timedelta(days=3), {"request": 30, "decline": 180}) == {"request": 1, "decline": 0}
    assert [r.correlation_id for r in episodes.requests()] == ["new"]


# ------------------------------------------------------------- consolidation

RECORDS = [  # newest first, as SqliteEpisodeMemory.requests returns them
    record("r6", "completed", 6, intent="ask", request="What does @Primary do?",
           message="It marks the preferred bean.", reflection_score=0.95),
    record("r5", "completed", 5, intent="ask", request="What is a bean?", message="An object.", reflection_score=0.5),
    record("r4", "clarification_needed", 4, intent="remove_example", request="Remove that example please."),
    record("r3", "clarification_needed", 3, intent="remove_example", request="Remove that example."),
    record("r2", "rejected", 2, intent="add_example", role="viewer", request="Add Q: What is AOP? A: Proxies.",
           message="role 'viewer' may not add_example (requires editor)"),
    record("r1", "completed", 1, intent="dataset_stats", request="How many examples?", message="10 training examples"),
]


def hashing_cluster(texts):
    return cluster_requests(texts, load_memory_config(OpsSettings().config_dir, env={}), env={})


def test_consolidation_groups_failures_and_proposes_cases():
    result = consolidate(RECORDS, hashing_cluster, days=30, now=T0)

    assert result.golden_candidates["cases"] == [
        {"id": "candidate_1", "role": "admin", "request": "Remove that example please.", "approve": False,
         "expect": {"outcomes": ["clarification_needed"], "dataset_unchanged": True}},
        {"id": "candidate_2", "role": "viewer", "request": "Add Q: What is AOP? A: Proxies.", "approve": False,
         "expect": {"outcomes": ["rejected"], "dataset_unchanged": True}},
    ]
    assert result.golden_candidates["observed"]["candidate_1"]["correlation_ids"] == ["r4", "r3"]
    assert result.training_candidates == [{"messages": [
        {"role": "user", "content": "What does @Primary do?"},
        {"role": "assistant", "content": "It marks the preferred bean."}]}]
    assert "6 request(s), 3 not completed, 2 group(s) of similar failures, 1 training candidate(s)." in result.report
    assert "1. 2 x {'clarification_needed': 2}: `Remove that example please.`" in result.report


def test_consolidation_files(tmp_path):
    paths = write(consolidate(RECORDS, hashing_cluster, days=30, now=T0), tmp_path / "out")
    report, golden, training = paths
    assert golden.read_text().startswith("# Candidate eval cases from failed requests")
    assert yaml.safe_load(golden.read_text())["cases"][0]["id"] == "candidate_1"
    assert [json.loads(line) for line in training.read_text().splitlines()][0]["messages"][0]["role"] == "user"
    assert report.read_text().startswith("# Consolidation report (2026-10-01, last 30 days)")


def test_golden_candidates_are_valid_eval_cases():
    """A reviewer only edits `expect`; the case shape must already load as a golden case."""
    pytest.importorskip("crewai")
    from ops_crew.evals.runner import GoldenCase

    for case in consolidate(RECORDS, hashing_cluster, days=30, now=T0).golden_candidates["cases"]:
        GoldenCase.model_validate(case)


def test_consolidate_cli(tmp_path, monkeypatch, capsys):
    from ops_crew.consolidate import main

    monkeypatch.setenv("OPS_STATE_DIR", str(tmp_path))
    assert main([]) == 0 and "no episodes yet" in capsys.readouterr().out
    now = datetime.now(timezone.utc)  # the CLI uses the real clock
    episodes = SqliteEpisodeMemory(tmp_path / "memory.db")
    episodes.record_request(replace(RECORDS[3], correlation_id="recent", ts=now - timedelta(days=1)))
    episodes.record_request(replace(RECORDS[2], correlation_id="ancient", ts=now - timedelta(days=400)))
    assert main(["--days", "1000", "--prune"]) == 0
    out = capsys.readouterr().out
    assert "wrote" in out and (tmp_path / "consolidation" / "report.md").exists()
    assert "pruned: 1 request episode(s), 0 decline episode(s)" in out  # retention: 30 days
    assert [r.correlation_id for r in episodes.requests()] == ["recent"]


# ------------------------------------------------------------------- flow

@pytest.fixture
def flow():
    pytest.importorskip("crewai")
    from ops_crew import flow as module

    return module


def remove_that(confidence=0.4):
    from ops_fakes import make_run

    return make_run("remove_example", confidence=confidence, clarifying_question="Which example do you want to remove?")


def test_an_answer_to_a_clarifying_question_continues_the_request(tmp_path, flow):
    from ops_fakes import FakeProposer, RecordingApprovals, build_deps, make_run

    proposer = FakeProposer(remove_that(), make_run("remove_example", params={"example_id": SCOPES_ID},
                                                    related_ids=[SCOPES_ID]))
    deps, sink = build_deps(tmp_path, proposer, RecordingApprovals(True), memory=True)
    first = flow.run_request(deps, "Remove that example.", flow.Role.ADMIN)
    assert (first.outcome, first.correlation_id) == ("clarification_needed", "cid000")

    second = flow.run_request(deps, "the one about bean scopes", flow.Role.ADMIN, reply_to="cid000")
    joined = ("Earlier request: Remove that example.\nClarifying question: Which example do you want to remove?\n"
              "Answer: the one about bean scopes")
    assert proposer.calls[1] == (joined, "cid001") and second.request == joined
    assert second.outcome == "completed" and deps.repo.get(SCOPES_ID) is None
    assert deps.episodes.get_request("cid001").reply_to == "cid000"
    assert [r["reply_to"] for r in sink.records if r["step"] == "intake" and "reply_to" in r] == ["cid000"]


@pytest.mark.parametrize("setup, reply_to, error", [
    ("none", "nope", "nothing to reply to: no request nope in memory"),
    ("completed", "cid000", "request cid000 did not ask a question (its outcome was completed)"),
    ("expired", "cid000", "the question from request cid000 has expired (older than 60 minutes); send a new request"),
])
def test_unacceptable_replies_are_rejected_before_the_crew(tmp_path, flow, setup, reply_to, error):
    from ops_fakes import FakeProposer, build_deps, make_run

    first = make_run("dataset_stats") if setup == "completed" else remove_that()
    proposer = FakeProposer(first)
    deps, _ = build_deps(tmp_path, proposer, memory=True)
    if setup != "none":
        flow.run_request(deps, "Remove that example.", flow.Role.ADMIN)
    if setup == "expired":
        deps.now = lambda: T0 + timedelta(hours=2)
    result = flow.run_request(deps, "the bean scopes one", flow.Role.ADMIN, reply_to=reply_to)
    assert (result.outcome, result.message) == ("rejected", f"invalid request: {error}")
    assert len(proposer.calls) == (0 if setup == "none" else 1)
    assert deps.episodes.get_request(result.correlation_id) is None  # never got past intake: not remembered


def test_replies_need_episodic_memory(tmp_path, flow):
    from ops_fakes import FakeProposer, build_deps

    deps, _ = build_deps(tmp_path, FakeProposer())
    result = flow.run_request(deps, "the bean scopes one", flow.Role.ADMIN, reply_to="cid000")
    assert result.message == "invalid request: replies need episodic memory (OPS_MEMORY=true)"


def test_a_declined_action_needs_a_human_next_time_too(tmp_path, flow):
    """The first add carried reviewer issues, so it was approved by a human, who declined. The same
    add without issues would normally go through; episodic memory sends it to a human again."""
    from ops_fakes import FakeProposer, RecordingApprovals, build_deps, make_run

    params = {"question": NEW_Q, "answer": NEW_A}
    approvals = RecordingApprovals(False)
    proposer = FakeProposer(make_run("add_example", params=params, issues=["unsure about the answer"]),
                            make_run("add_example", params=params))
    deps, _ = build_deps(tmp_path, proposer, approvals, memory=True)
    assert flow.run_request(deps, f"add Q: {NEW_Q} A: {NEW_A}", flow.Role.EDITOR).outcome == "declined"
    deps.now = lambda: T0 + timedelta(days=1)
    again = flow.run_request(deps, f"please add Q: {NEW_Q} A: {NEW_A}", flow.Role.EDITOR)

    assert again.outcome == "declined" and len(deps.repo.examples()) == 10
    assert approvals.requests[1].reasons == ("a human declined this exact action on 2026-10-01 12:00 UTC",)


def test_every_finished_request_is_remembered_and_memory_failures_do_not_break_it(tmp_path, flow):
    from ops_fakes import FakePlanner, FakeProposer, build_deps, make_plan, make_run

    planner = FakePlanner(make_plan(("dataset_stats", "How many examples are there?"),
                                    ("ask", "What is constructor injection?")))
    proposer = FakeProposer(make_run("dataset_stats"), make_run("ask", answer="Dependencies via the constructor."))
    deps, _ = build_deps(tmp_path, proposer, planner=planner, memory=True)
    result = flow.run_request(deps, "How many examples are there? What is constructor injection?", flow.Role.VIEWER)
    stored = deps.episodes.get_request(result.correlation_id)
    assert (stored.outcome, stored.plan_intents, stored.step_outcomes) == (
        "completed", ("dataset_stats", "ask"), ("completed", "completed"))

    class Broken:
        def record_request(self, record):
            raise OSError("disk full")

    (tmp_path / "b").mkdir()
    deps2, sink = build_deps(tmp_path / "b", FakeProposer(make_run("dataset_stats")))
    deps2.episodes = Broken()
    assert flow.run_request(deps2, "How many examples?", flow.Role.VIEWER).outcome == "completed"
    assert [r["error"] for r in sink.records if r["step"] == "memory_write_failed"] == ["disk full"]
