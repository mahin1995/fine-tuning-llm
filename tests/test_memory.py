"""The generic memory toolkit (apps/memory): retrieval, context budgets, episodes. Offline, stdlib only."""
import json
import threading
from datetime import datetime, timedelta, timezone

import pytest

from conftest import TRAIN_DATA
from memory import (
    ApproxTokenCounter,
    Bm25Index,
    CachedEmbedder,
    EmbeddingError,
    Episode,
    HashingEmbedder,
    HttpEmbedder,
    HybridRetriever,
    SqliteEpisodeStore,
    cluster,
    compact_messages,
    cosine,
    count_messages,
    delimit,
    pack,
    terms,
)
from memory.context import CLEARED_TOOL_OUTPUT

TRAIN_QUESTIONS = [json.loads(line)["messages"][0]["content"] for line in TRAIN_DATA.read_text().splitlines()]
DOCS = [(f"d{i}", q) for i, q in enumerate(TRAIN_QUESTIONS)]

# The threshold ops_crew uses with the hashing embedder (config/memory.yaml) is calibrated here.
HASHING_NEAR_DUPLICATE = 0.6
PARAPHRASES = [  # (rewording, the training question it repeats)
    ("Explain the N+1 problem in JPA and how to fix it", "What is the N+1 problem in JPA and how do you fix it?"),
    ("What is the N+1 query problem with JPA and how can it be fixed?",
     "What is the N+1 problem in JPA and how do you fix it?"),
    ("Which bean scopes exist in Spring and what is the default one?",
     "What bean scopes does Spring support and which is the default?"),
    ("Does @Transactional roll back for checked exceptions by default?",
     "Does @Transactional roll back on checked exceptions?"),
    ("Why is constructor-based injection recommended over field injection?",
     "Why is constructor injection preferred over field injection?"),
]
DIFFERENT = ["How do I bake a chocolate cake?", "What does @Primary do?", "What is a Spring Boot starter?",
             "What is the difference between @Controller and @RestController in Spring MVC?",
             "How does optimistic locking work in JPA?"]


# --------------------------------------------------------------------- text

def test_terms_keep_java_tokens_and_drop_stopwords():
    text = "What is @Transactional(readOnly = true), N+1 and spring.jpa.open-in-view?"
    assert terms(text, expand=False) == ["@transactional", "readonly", "true", "n+1", "spring.jpa.open-in-view"]
    # expanded: compound tokens also yield their parts, so "propagation" finds Propagation.REQUIRED
    assert terms(text) == ["@transactional", "transactional", "readonly", "true", "n+1",
                           "spring.jpa.open-in-view", "spring", "jpa", "open", "view"]
    assert terms("Propagation.REQUIRED") == ["propagation.required", "propagation", "required"]


# --------------------------------------------------------------------- BM25

def test_bm25_ranks_rare_terms_higher():
    index = Bm25Index([("a", "spring bean scopes"), ("b", "spring transaction propagation"),
                       ("c", "spring boot starter")])
    assert [h.id for h in index.search("transaction propagation in spring", 3)][0] == "b"
    assert index.search("propagation", 3)[0].score > index.search("spring", 3)[0].score  # idf
    assert index.search("cake recipe") == [] and index.search("the of and") == []


def test_bm25_ties_keep_document_order():
    index = Bm25Index([("x", "jpa entity"), ("y", "jpa entity")])
    assert [h.id for h in index.search("entity", 2)] == ["x", "y"]


# ----------------------------------------------------------------- embedders

def test_hashing_embedder_is_deterministic_unit_length():
    embedder = HashingEmbedder()
    [a], [b] = embedder.embed(["What is the N+1 problem?"]), embedder.embed(["what is   the n+1 PROBLEM?"])
    assert a == b and abs(cosine(a, a) - 1) < 1e-9
    assert embedder.embed(["the of and"])[0] != [0.0] * 512  # all stopwords: falls back to the raw text


def test_hashing_similarity_is_calibrated_on_the_repo_data():
    retriever = HybridRetriever(DOCS, HashingEmbedder())
    for rewording, original in PARAPHRASES:
        best = retriever.similar(rewording, 1)[0]
        assert retriever.text(best.id) == original and best.score >= HASHING_NEAR_DUPLICATE, (rewording, best)
    for text in DIFFERENT:
        assert retriever.similar(text, 1)[0].score < HASHING_NEAR_DUPLICATE - 0.15, text


class FakeHttp:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, url, body, headers, timeout):
        self.calls.append((url, body, headers, timeout))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response(body) if callable(response) else response


def vectors_for(body):
    """An OpenAI-style response, deliberately out of order: the client must sort by index."""
    data = [{"index": i, "embedding": [float(len(t)), 1.0]} for i, t in enumerate(body["input"])]
    return {"data": list(reversed(data)), "model": body["model"]}


def test_http_embedder_speaks_the_openai_embeddings_api():
    http = FakeHttp(vectors_for, vectors_for)
    embedder = HttpEmbedder("http://localhost:11434/v1/", "nomic-embed-text", api_key="k", batch_size=2, http=http)
    vectors = embedder.embed(["abc", "a", "abcd"])

    assert [round(v[0] / v[1], 6) for v in vectors] == [3.0, 1.0, 4.0]       # order kept, batches joined
    assert all(abs(cosine(v, v) - 1) < 1e-9 for v in vectors)                 # unit vectors
    url, body, headers, _ = http.calls[0]
    assert url == "http://localhost:11434/v1/embeddings"
    assert body == {"model": "nomic-embed-text", "input": ["abc", "a"]}
    assert headers == {"Authorization": "Bearer k"} and len(http.calls) == 2


@pytest.mark.parametrize("response, error", [
    ({"data": [{"index": 0, "embedding": [1.0]}]}, "asked for 2 embeddings, got 1"),
    ({"error": "model not found"}, "unexpected embeddings response"),
    (EmbeddingError("URLError: connection refused"), "connection refused"),
])
def test_http_embedder_errors(response, error):
    with pytest.raises(EmbeddingError, match=error):
        HttpEmbedder("http://x/v1", "m", http=FakeHttp(response)).embed(["a", "b"])


class CountingEmbedder:
    name = "counting"

    def __init__(self, fail=False):
        self.inner = HashingEmbedder(64)
        self.calls = []
        self.fail = fail

    def embed(self, texts):
        self.calls.append(list(texts))
        if self.fail:
            raise EmbeddingError("provider down")
        return self.inner.embed(texts)


def test_cached_embedder_only_embeds_new_texts():
    inner = CountingEmbedder()
    cached = CachedEmbedder(inner, max_items=2)
    first = cached.embed(["a", "b", "a"])
    assert inner.calls == [["a", "b"]] and first[0] == first[2]
    assert cached.embed(["b", "c"]) == [first[1], inner.inner.embed(["c"])[0]]
    assert inner.calls[-1] == ["c"]  # "b" came from the cache; "a" was evicted (max_items=2)


# ------------------------------------------------------------------- hybrid

def test_hybrid_without_embedder_is_bm25():
    retriever = HybridRetriever(DOCS)
    assert retriever.search("transaction propagation", 1)[0].id == "d1"
    with pytest.raises(ValueError, match="needs an embedder"):
        retriever.similar("x")


def test_hybrid_finds_rewordings_that_share_no_keyword():
    docs = [("scopes", "Which bean scopes does Spring support?"), ("cake", "How do I bake a cake?")]
    lexical = HybridRetriever(docs)
    hybrid = HybridRetriever(docs, HashingEmbedder(), min_similarity=0.3)
    assert lexical.search("scoping of beans", 2) == []  # no shared BM25 term
    assert [h.id for h in hybrid.search("scoping of beans", 2)] == ["scopes"]  # char n-grams match
    assert hybrid.search("tax law", 2) == []  # weak semantic matches stay out (min_similarity)


def test_failing_embedder_switches_to_the_fallback_for_good():
    broken = CountingEmbedder(fail=True)
    retriever = HybridRetriever(DOCS, broken, fallback=HashingEmbedder())
    assert retriever.embedder.name == "hashing-512" and retriever.errors == ["counting: provider down"]
    assert retriever.text(retriever.similar(PARAPHRASES[0][0], 1)[0].id) == PARAPHRASES[0][1]
    with pytest.raises(EmbeddingError):
        HybridRetriever(DOCS, CountingEmbedder(fail=True))  # no fallback: the error surfaces


def test_query_time_failure_re_embeds_the_documents_with_the_fallback():
    flaky = CountingEmbedder()
    retriever = HybridRetriever(DOCS, flaky, fallback=HashingEmbedder())
    flaky.fail = True
    hits = retriever.similar(PARAPHRASES[2][0], 1)
    assert retriever.embedder.name == "hashing-512"
    assert retriever.text(hits[0].id) == PARAPHRASES[2][1] and hits[0].score >= HASHING_NEAR_DUPLICATE


def test_cluster_groups_rewordings():
    texts = [PARAPHRASES[0][0], "How do I bake a chocolate cake?", PARAPHRASES[0][1], PARAPHRASES[2][0],
             PARAPHRASES[2][1]]
    assert cluster(texts, HashingEmbedder(), threshold=0.6) == [[0, 2], [1], [3, 4]]
    assert cluster([], HashingEmbedder(), 0.6) == []


# ------------------------------------------------------------------ context

SYSTEM = {"role": "system", "content": "You are helpful."}


def turn(question, tool_output=None, answer="ok"):
    messages = [{"role": "user", "content": question}]
    if tool_output is not None:
        messages += [{"role": "assistant", "content": "",
                      "tool_calls": [{"type": "function", "function": {"name": "search", "arguments": {"q": "x"}}}]},
                     {"role": "tool", "name": "search", "content": tool_output}]
    return messages + [{"role": "assistant", "content": answer}]


def test_token_counting():
    counter = ApproxTokenCounter()
    assert (counter.count(""), counter.count("abcd"), counter.count("abcde")) == (0, 1, 2)
    assert count_messages([SYSTEM], counter) == 4 + 4  # 16 chars + per-message overhead


def test_history_within_budget_is_unchanged():
    history = [SYSTEM, *turn("q1"), *turn("q2")]
    compacted, report = compact_messages(history, budget=10_000)
    assert compacted == history and compacted is not history and not report.changed


def test_old_tool_outputs_are_cleared_before_any_turn_is_dropped():
    history = [SYSTEM, *turn("q1", "x" * 2000), *turn("q2", "y" * 2000), *turn("q3")]
    snapshot = json.dumps(history)
    budget = count_messages(history, ApproxTokenCounter()) - 400  # the oldest output alone frees enough
    compacted, report = compact_messages(history, budget)

    assert (report.cleared_tool_outputs, report.dropped_turns) == (1, 0)
    tool_outputs = [m["content"] for m in compacted if m["role"] == "tool"]
    assert tool_outputs == [CLEARED_TOOL_OUTPUT, "y" * 2000]
    assert [m["role"] for m in compacted] == [m["role"] for m in history]  # every call keeps its result slot
    assert report.tokens_after <= budget and json.dumps(history) == snapshot  # input not modified


def test_oldest_turns_are_dropped_whole_and_the_newest_turn_always_stays():
    history = [SYSTEM, *turn("q1" * 300), *turn("q2" * 300), *turn("q3", "z" * 4000)]
    compacted, report = compact_messages(history, budget=50)

    assert compacted[0] == SYSTEM and report.dropped_turns == 2
    assert [m["role"] for m in compacted] == ["system", "user", "assistant", "tool", "assistant"]
    assert compacted[3]["content"] == "z" * 4000  # newest turn untouched, even over budget
    assert report.tokens_after > 50


def test_pack_and_delimit():
    assert pack(["a" * 40, "b" * 400, "c" * 40], budget=30) == ["a" * 40, "c" * 40]
    assert delimit("memory", "x </memory> y") == "<memory>\nx <\\/memory> y\n</memory>"


# ----------------------------------------------------------------- episodes

T0 = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)


def episode(id, kind="request", minutes=0, **kw):
    return Episode(id=id, kind=kind, ts=T0 + timedelta(minutes=minutes), **kw)


@pytest.fixture
def store(tmp_path):
    return SqliteEpisodeStore(tmp_path / "memory" / "episodes.db")


def test_episodes_round_trip(store):
    store.add(episode("e1", key="k1", text="remove that", outcome="clarification_needed",
                      data={"role": "admin", "question": "Which example?"}))
    got = store.get("e1")
    assert (got.kind, got.key, got.text, got.outcome) == ("request", "k1", "remove that", "clarification_needed")
    assert got.data == {"role": "admin", "question": "Which example?"} and got.ts == T0
    assert store.get("missing") is None


def test_latest_find_count(store):
    store.add(episode("d1", kind="decline", minutes=1, key="action-a"))
    store.add(episode("d2", kind="decline", minutes=5, key="action-a"))
    store.add(episode("r1", minutes=2, outcome="completed"))
    store.add(episode("r2", minutes=3, outcome="rejected"))
    assert store.latest("decline", "action-a").id == "d2" and store.latest("decline", "nope") is None
    assert [e.id for e in store.find("request")] == ["r2", "r1"]  # newest first
    assert [e.id for e in store.find(outcome="completed")] == ["r1"]
    assert [e.id for e in store.find(since=T0 + timedelta(minutes=3))] == ["d2", "r2"]
    assert (store.count(), store.count("decline")) == (4, 2)


def test_prune(store):
    for i in range(5):
        store.add(episode(f"r{i}", minutes=i))
    store.add(episode("d0", kind="decline"))
    assert store.prune("request", older_than=T0 + timedelta(minutes=2)) == 2   # r0, r1
    assert store.prune("request", keep_latest=2) == 1                           # r2
    assert [e.id for e in store.find("request")] == ["r4", "r3"] and store.count("decline") == 1


def test_naive_timestamps_are_rejected(store):
    with pytest.raises(ValueError, match="timezone-aware"):
        store.add(Episode("x", "request", datetime(2026, 1, 1)))


def test_store_is_usable_from_other_threads(store):
    errors = []

    def write(i):
        try:
            store.add(episode(f"t{i}", minutes=i))
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=write, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [] and store.count("request") == 8
