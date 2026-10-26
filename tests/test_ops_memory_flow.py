"""ops_crew semantic memory and shared working state through the real flow and real CrewAI tools."""
import json

import pytest

pytest.importorskip("crewai")

from ops_crew.crew.tools import ReadOnlyDataset, ToolTrace, build_tools  # noqa: E402
from ops_crew.domain.repository import example_id_for  # noqa: E402
from ops_crew.flow import run_request  # noqa: E402
from ops_crew.schemas import ActionProposal, Role  # noqa: E402
from ops_fakes import FakeProposer, FakeRefiner, RecordingApprovals, build_deps, copy_datasets, make_run  # noqa: E402

SCOPES_ID = example_id_for("What bean scopes does Spring support and which is the default?")
N_PLUS_ONE_ID = example_id_for("What is the N+1 problem in JPA and how do you fix it?")
PARAPHRASE = "Which bean scopes exist in Spring and what is the default one?"
ANSWER = "singleton is the default; prototype, request, session, application and websocket also exist."
REQUEST = f"Add this example. Q: {PARAPHRASE} A: {ANSWER}"


def steps(sink):
    return [r["step"] for r in sink.records]


def paraphrase_run():
    return make_run("add_example", params={"question": PARAPHRASE, "answer": ANSWER})


@pytest.mark.parametrize("approve, outcome, examples", [(False, "declined", 10), (True, "completed", 11)])
def test_near_duplicate_needs_a_human(tmp_path, approve, outcome, examples):
    approvals = RecordingApprovals(approve)
    deps, sink = build_deps(tmp_path, FakeProposer(paraphrase_run()), approvals, memory=True)
    result = run_request(deps, REQUEST, Role.EDITOR)

    assert result.outcome == outcome and len(deps.repo.examples()) == examples
    [req] = approvals.requests
    assert req.reasons[0].startswith(f"similar to existing example {SCOPES_ID} (score 0.6")
    [record] = [r for r in sink.records if r["step"] == "similarity"]
    assert (record["closest_example"], record["embedder"], record["error"]) == (SCOPES_ID, "hashing-512", None)


def test_without_memory_the_near_duplicate_goes_through(tmp_path):
    """OPS_MEMORY=false: only exact duplicates are caught, as before Level 2."""
    approvals = RecordingApprovals(False)
    deps, sink = build_deps(tmp_path, FakeProposer(paraphrase_run()), approvals)
    result = run_request(deps, REQUEST, Role.EDITOR)
    assert result.outcome == "completed" and approvals.requests == [] and "similarity" not in steps(sink)


def test_a_failing_similarity_check_asks_a_human(tmp_path):
    class Broken:
        def report(self, text):
            raise OSError("disk unavailable")

    approvals = RecordingApprovals(False)
    deps, _ = build_deps(tmp_path, FakeProposer(paraphrase_run()), approvals)
    deps.similarity = Broken()
    assert run_request(deps, REQUEST, Role.EDITOR).outcome == "declined"
    assert approvals.requests[0].reasons == (
        "similarity check unavailable (OSError: disk unavailable); check for near duplicates by hand",)


def test_an_id_no_tool_returned_is_sent_back_for_correction(tmp_path):
    """Shared working state: the researcher's findings name the N+1 example, but no tool returned it."""
    findings = json.dumps({"summary": "found it", "related_example_ids": [N_PLUS_ONE_ID]})
    claimed_only = make_run("remove_example", params={"example_id": N_PLUS_ONE_ID}, research_context=findings)
    refiner = FakeRefiner(repairs=[ActionProposal(intent="remove_example", rationale="not confirmed", confidence=0.4,
                                                  clarifying_question="Which example do you want to remove?")])
    deps, _ = build_deps(tmp_path, FakeProposer(claimed_only), RecordingApprovals(True), refiner=refiner)
    result = run_request(deps, "remove the N+1 example", Role.ADMIN)

    assert result.outcome == "clarification_needed" and len(deps.repo.examples()) == 10
    [req] = refiner.repair_calls
    assert req.problems == [f"params.example_id {N_PLUS_ONE_ID} was not returned by any dataset tool (returned: "
                            "none); use one of those, or leave it empty and set clarifying_question"]


# -------------------------------------------------------------------- tools

def data(output):
    return json.loads(output.split("\n")[1])


@pytest.fixture
def tools(tmp_path):
    deps, _ = build_deps(tmp_path, None, memory=True)
    trace = ToolTrace()
    names = ["search_training_data", "find_similar", "find_duplicate", "verified_facts"]
    return dict(zip(names, build_tools(names, "researcher", ReadOnlyDataset(deps.repo, deps.similarity), trace))), trace


def test_search_and_find_similar_use_meaning(tools):
    tools, trace = tools
    assert data(tools["search_training_data"].run(query="scoping of beans", limit=1))[0]["example_id"] == SCOPES_ID
    similar = data(tools["find_similar"].run(question=PARAPHRASE, limit=2))
    assert similar["similar"][0]["example_id"] == SCOPES_ID and similar["similar"][0]["score"] >= 0.6
    assert set(similar) == {"similar", "eval_similarity"}  # eval questions are scored, never shown
    assert trace.records[1].facts["similar"][0] == {"example_id": SCOPES_ID, "score": similar["similar"][0]["score"]}


def test_verified_facts_show_what_tools_returned(tools):
    tools, trace = tools
    tools["search_training_data"].run(query="N+1 JPA", limit=1)
    tools["find_duplicate"].run(question=PARAPHRASE)
    facts = data(tools["verified_facts"].run())
    assert facts["example_ids"] == [N_PLUS_ONE_ID] and facts["duplicate_of"] is None
    assert [c.tool for c in trace.records] == ["search_training_data", "find_duplicate", "verified_facts"]


def test_find_similar_without_memory_reports_an_error(tmp_path):
    [tool] = build_tools(["find_similar"], "researcher", ReadOnlyDataset(copy_datasets(tmp_path)), ToolTrace())
    assert data(tool.run(question=PARAPHRASE)) == {
        "error": "RuntimeError: similarity search is not configured (OPS_MEMORY=false)"}
