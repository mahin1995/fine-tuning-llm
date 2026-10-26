"""ops_crew semantic memory and shared working state, without CrewAI: the similarity adapter,
the policy rule for near duplicates, and the tool-facts blackboard."""
import pytest

pytest.importorskip("pydantic_settings")  # ops_crew deps (apps/requirements.txt)

from memory import EmbeddingError, HashingEmbedder, HttpEmbedder  # noqa: E402
from ops_crew.crew.config import ConfigError  # noqa: E402
from ops_crew.domain.policy import Outcome, authorize, similarity_reasons  # noqa: E402
from ops_crew.domain.ports import SimilarExample, SimilarityReport  # noqa: E402
from ops_crew.domain.repository import example_id_for  # noqa: E402
from ops_crew.domain.working_memory import VerifiedFacts  # noqa: E402
from ops_crew.memory_adapters import DatasetSimilarity, Thresholds, build_embedder, load_memory_config  # noqa: E402
from ops_crew.schemas import ActionProposal, Role, ToolCallRecord  # noqa: E402
from ops_crew.settings import OpsSettings  # noqa: E402
from ops_fakes import copy_datasets  # noqa: E402

CONFIG_DIR = OpsSettings().config_dir
SCOPES_ID = example_id_for("What bean scopes does Spring support and which is the default?")
PARAPHRASE = "Which bean scopes exist in Spring and what is the default one?"
NEAR_EVAL = "What's the difference between the @Controller and @RestController annotations?"
NEW_Q = "What is the difference between @Value and @ConfigurationProperties?"
HASHING = {"hashing": Thresholds(near_duplicate=0.6, near_eval=0.6),
           "openai_compatible": Thresholds(near_duplicate=0.9, near_eval=0.9)}


@pytest.fixture
def repo(tmp_path):
    return copy_datasets(tmp_path)


@pytest.fixture
def similarity(repo):
    return DatasetSimilarity(repo, HashingEmbedder(), HASHING)


# ------------------------------------------------------------------- config

def test_repo_memory_config_defaults_to_offline_hashing():
    config = load_memory_config(CONFIG_DIR, env={})
    assert config.embeddings.provider == "hashing" and config.embeddings.base_url == "http://localhost:11434/v1"
    assert config.similarity["hashing"] == Thresholds(near_duplicate=0.6, near_eval=0.6, cluster=0.5)
    assert config.episodes.retention_days == {"request": 30, "decline": 180}
    assert isinstance(build_embedder(config.embeddings, env={}), HashingEmbedder)


def test_memory_config_from_env():
    env = {"OPS_EMBED_PROVIDER": "openai_compatible", "OPS_EMBED_BASE_URL": "http://gpu:8000/v1",
           "OPS_EMBED_API_KEY": "k"}
    config = load_memory_config(CONFIG_DIR, env=env)
    assert (config.embeddings.provider, config.embeddings.base_url) == ("openai_compatible", "http://gpu:8000/v1")
    embedder = build_embedder(config.embeddings, env=env)
    assert embedder.name == "nomic-embed-text" and embedder.inner.url == "http://gpu:8000/v1/embeddings"


@pytest.mark.parametrize("mutate, error", [
    (lambda s: s.replace("${OPS_EMBED_PROVIDER:-hashing}", "word2vec"), "provider"),
    (lambda s: s.replace("  hashing: {near_duplicate: 0.6, near_eval: 0.6, cluster: 0.5}\n", ""),
     "thresholds needed for 'hashing'"),
    (lambda s: s.replace("near_duplicate: 0.6", "near_duplicate: 1.5"), "less than or equal to 1"),
])
def test_memory_config_errors_fail_fast(tmp_path, mutate, error):
    (tmp_path / "memory.yaml").write_text(mutate((CONFIG_DIR / "memory.yaml").read_text()))
    with pytest.raises(ConfigError, match=error):
        load_memory_config(tmp_path, env={})


# ------------------------------------------------------------- similarity

def test_report_finds_near_duplicates_and_near_eval_questions(similarity):
    report = similarity.report(PARAPHRASE)
    assert report.closest_example.example_id == SCOPES_ID and report.closest_example.score >= 0.6
    assert (report.near_duplicate_threshold, report.embedder, report.error) == (0.6, "hashing-512", None)
    assert similarity.report(NEAR_EVAL).eval_score >= 0.6
    unrelated = similarity.report(NEW_Q)
    assert unrelated.closest_example.score < 0.6 and unrelated.eval_score < 0.6


def test_index_follows_dataset_changes(repo, similarity):
    question = "What does @Primary do?"
    assert similarity.report(question).closest_example.score < 0.6
    repo.append(question, "It marks the bean to inject when several beans of the same type match.")
    report = similarity.report(question)
    assert report.closest_example.example_id == example_id_for(question)
    assert report.closest_example.score > 0.99


def test_search_matches_meaning_as_well_as_keywords(repo, similarity):
    assert similarity.search("scoping of beans", 1) == [SCOPES_ID]  # no BM25 term in common
    assert repo.search("scoping of beans", 1) == []                 # the old keyword search misses it


def test_provider_failure_falls_back_to_hashing_and_its_thresholds(repo):
    def down(url, body, headers, timeout):
        raise EmbeddingError("URLError: connection refused")

    similarity = DatasetSimilarity(repo, HttpEmbedder("http://ollama:11434/v1", "nomic-embed-text", http=down),
                                   HASHING)
    report = similarity.report(PARAPHRASE)
    assert (report.embedder, report.near_duplicate_threshold) == ("hashing-512", 0.6)  # not 0.9
    assert report.closest_example.example_id == SCOPES_ID
    assert similarity.errors == ["nomic-embed-text: URLError: connection refused"] * 2  # train and eval index


# ------------------------------------------------------------------- policy

def report(score=0.7, eval_score=0.1, error=None):
    return SimilarityReport(SimilarExample(SCOPES_ID, score), eval_score, 0.6, 0.6, "hashing-512", error)


def add(question):
    return ActionProposal(intent="add_example", params={"question": question, "answer": "A" * 40}, rationale="r",
                          confidence=0.9)


def test_similarity_reasons():
    assert similarity_reasons(report(0.67)) == [f"similar to existing example {SCOPES_ID} (score 0.67)"]
    assert similarity_reasons(report(0.2, eval_score=0.86)) == [
        "close to an eval question (score 0.86): possible leakage"]
    assert similarity_reasons(report(0.59, eval_score=0.59)) == []
    assert similarity_reasons(SimilarityReport.unavailable("IndexError: boom")) == [
        "similarity check unavailable (IndexError: boom); check for near duplicates by hand"]


def test_near_duplicates_need_approval_exact_duplicates_stay_rejected(repo):
    near = authorize(add(PARAPHRASE), Role.EDITOR, repo, similarity=report(0.67))
    assert (near.outcome, near.reasons) == (Outcome.NEEDS_APPROVAL,
                                            [f"similar to existing example {SCOPES_ID} (score 0.67)"])
    exact = authorize(add("What bean scopes does Spring support and which is the default?"), Role.EDITOR, repo,
                      similarity=report(1.0))
    assert exact.outcome == Outcome.DENY  # a rule, not a fuzzy signal
    clear = authorize(add(NEW_Q), Role.EDITOR, repo, similarity=report(0.3))
    assert clear.outcome == Outcome.ALLOW


# ------------------------------------------------------------ working memory

def test_verified_facts_fold_what_the_tools_returned():
    calls = [
        ToolCallRecord(agent="researcher", tool="search_training_data", arguments={}, facts={"example_ids": ["a1"]}),
        ToolCallRecord(agent="researcher", tool="find_duplicate", arguments={},
                       facts={"duplicate_of": None, "example_ids": []}),
        ToolCallRecord(agent="researcher", tool="check_eval_overlap", arguments={}, facts={"eval_overlap": False}),
        ToolCallRecord(agent="researcher", tool="find_similar", arguments={},
                       facts={"example_ids": ["b2", "a1"], "eval_similarity": 0.12,
                              "similar": [{"example_id": "b2", "score": 0.5}, {"example_id": "a1", "score": 0.8}]}),
        ToolCallRecord(agent="reviewer", tool="get_example", arguments={}, facts={}),  # a failed call adds nothing
    ]
    facts = VerifiedFacts.from_tool_calls(calls)
    assert facts.example_ids == {"a1", "b2"} and facts.duplicate_of is None and facts.eval_overlap is False
    assert facts.as_dict() == {"example_ids": ["a1", "b2"], "duplicate_of": None, "eval_overlap": False,
                               "similar": [{"example_id": "a1", "score": 0.8}, {"example_id": "b2", "score": 0.5}],
                               "eval_similarity": 0.12}
