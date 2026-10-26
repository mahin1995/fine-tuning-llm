"""Retrieval: BM25 (lexical), embeddings (semantic) and both fused with Reciprocal Rank Fusion.

    Bm25Index        Okapi BM25 over (id, text) documents. Build O(total terms), query
                     O(query terms x postings).
    Embedder         port: embed(texts) -> unit vectors. Adapters:
                       HashingEmbedder  char n-gram feature hashing: offline, deterministic,
                                        matches shared wording, not synonyms
                       HttpEmbedder     OpenAI-compatible POST /embeddings (Ollama, vLLM, OpenAI)
                       CachedEmbedder   memoises any embedder by text
    HybridRetriever  BM25 + vectors (brute-force cosine, O(N x d) per query; fine up to ~10k
                     documents, use an ANN index such as HNSW or pgvector beyond that). If the
                     embedder fails, it switches to the fallback embedder for good, so an index
                     never mixes vectors from two models.
"""
import hashlib
import heapq
import json
import math
import urllib.error
import urllib.request
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol

from memory.text import char_ngrams, normalize, terms


@dataclass(frozen=True)
class Hit:
    id: str
    score: float


class EmbeddingError(RuntimeError):
    """The embedding provider failed or returned something unusable."""


def unit(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm else list(vector)


def cosine(a: list[float], b: list[float]) -> float:
    """Dot product of unit vectors (every embedder here returns unit vectors)."""
    return sum(x * y for x, y in zip(a, b))


def _top(scored: Iterable[tuple[float, int]], k: int) -> list[tuple[float, int]]:
    """Highest scores first; ties keep document order."""
    return sorted(heapq.nlargest(k, scored, key=lambda s: (s[0], -s[1])), key=lambda s: (-s[0], s[1]))


# --------------------------------------------------------------------- BM25

class Bm25Index:
    def __init__(self, docs: Iterable[tuple[str, str]], k1: float = 1.5, b: float = 0.75):
        self.ids: list[str] = []
        self._lengths: list[int] = []
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)  # term -> [(doc, tf)]
        for doc_id, text in docs:
            counts = Counter(terms(text))
            index = len(self.ids)
            self.ids.append(doc_id)
            self._lengths.append(sum(counts.values()))
            for term, tf in counts.items():
                self._postings[term].append((index, tf))
        self._k1, self._b = k1, b
        self._avg_len = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0

    def _idf(self, term: str) -> float:
        df, n = len(self._postings[term]), len(self.ids)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))  # never negative

    def search(self, query: str, k: int = 5) -> list[Hit]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(terms(query)):
            if term not in self._postings:
                continue
            idf = self._idf(term)
            for index, tf in self._postings[term]:
                norm = 1 - self._b + self._b * self._lengths[index] / (self._avg_len or 1)
                scores[index] += idf * tf * (self._k1 + 1) / (tf + self._k1 * norm)
        return [Hit(self.ids[i], s) for s, i in _top(((s, i) for i, s in scores.items()), k)]


# ---------------------------------------------------------------- embedders

class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]:
        """One unit vector per text, same order. Raises EmbeddingError."""
        ...


class HashingEmbedder:
    """Signed feature hashing of the character 3-5-grams of a text's content words. Stopwords
    are dropped first, so question boilerplate ("what is the difference between") doesn't make
    unrelated questions look alike. No model, no network, the same vectors in every process
    (crc32, not Python's randomised hash). It catches rewordings that keep the key words (word
    order, plurals, typos, extra words); only a trained model knows synonyms."""

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for gram in char_ngrams(" ".join(terms(text, expand=False)) or normalize(text)):
            h = zlib.crc32(gram.encode("utf-8"))
            vector[h % self.dim] += 1.0 if h & 0x80000000 else -1.0
        return unit(vector)


# (url, json body, headers, timeout) -> parsed JSON response
Http = Callable[[str, dict, dict, float], dict]


def urllib_post(url: str, body: dict, headers: dict, timeout: float) -> dict:
    request = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise EmbeddingError(f"{type(e).__name__}: {e}") from e


class HttpEmbedder:
    """OpenAI-compatible embeddings endpoint: POST {base_url}/embeddings {model, input: [...]}.
    Works with Ollama (/v1, e.g. nomic-embed-text), vLLM and OpenAI."""

    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout: float = 30,
                 batch_size: int = 64, http: Http = urllib_post):
        self.url = base_url.rstrip("/") + "/embeddings"
        self.model = model
        self.name = model
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._timeout = timeout
        self._batch_size = batch_size
        self._http = http

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start:start + self._batch_size]
            response = self._http(self.url, {"model": self.model, "input": batch}, self._headers, self._timeout)
            try:
                data = sorted(response["data"], key=lambda d: d["index"])
                batch_vectors = [unit([float(x) for x in d["embedding"]]) for d in data]
            except (KeyError, TypeError, ValueError) as e:
                raise EmbeddingError(f"unexpected embeddings response: {type(e).__name__}: {e}") from e
            if len(batch_vectors) != len(batch):
                raise EmbeddingError(f"asked for {len(batch)} embeddings, got {len(batch_vectors)}")
            vectors.extend(batch_vectors)
        return vectors


class CachedEmbedder:
    """Memoises an embedder by text content, so re-indexing a dataset only embeds new texts."""

    def __init__(self, inner: Embedder, max_items: int = 50_000):
        self.inner = inner
        self.name = inner.name
        self._cache: dict[str, list[float]] = {}
        self._max_items = max_items

    def embed(self, texts: list[str]) -> list[list[float]]:
        keys = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts]
        missing: dict[str, str] = {}
        for key, text in zip(keys, texts):
            if key not in self._cache and key not in missing:
                missing[key] = text
        fresh = dict(zip(missing, self.inner.embed(list(missing.values())))) if missing else {}
        for key, vector in fresh.items():
            if len(self._cache) >= self._max_items:
                self._cache.pop(next(iter(self._cache)))  # drop the oldest entry
            self._cache[key] = vector
        return [fresh[key] if key in fresh else self._cache[key] for key in keys]


# -------------------------------------------------------------------- hybrid

class HybridRetriever:
    """BM25 plus (optionally) embeddings, fused with Reciprocal Rank Fusion:
    score(d) = sum over rankings of 1 / (rrf_k + rank). Without an embedder it is plain BM25.

    `min_similarity` keeps weak semantic matches out of search results (a vector index always
    returns *something*); `similar()` returns raw cosine scores for threshold decisions.
    `vector_texts` (same order as `docs`) embeds different text than BM25 indexes, e.g. BM25
    over a boosted "question question answer" and vectors over the plain question."""

    def __init__(self, docs: Iterable[tuple[str, str]], embedder: Embedder | None = None, *,
                 fallback: Embedder | None = None, min_similarity: float = 0.0, rrf_k: int = 60,
                 vector_texts: list[str] | None = None):
        self._docs = list(docs)
        self._texts = {doc_id: text for doc_id, text in self._docs}
        self._vector_texts = list(vector_texts) if vector_texts is not None else [t for _, t in self._docs]
        if len(self._vector_texts) != len(self._docs):
            raise ValueError("vector_texts must have one text per document")
        self.bm25 = Bm25Index(self._docs)
        self.embedder = embedder
        self._fallback = fallback
        self.min_similarity = min_similarity
        self._rrf_k = rrf_k
        self.errors: list[str] = []
        self._vectors: list[list[float]] | None = None
        if embedder is not None:
            self._vectors = self._with_fallback(lambda e: e.embed(self._vector_texts))

    def _with_fallback(self, fn):
        try:
            return fn(self.embedder)
        except EmbeddingError as e:
            if self._fallback is None or self.embedder is self._fallback:
                raise
            self.errors.append(f"{self.embedder.name}: {e}")
            self.embedder = self._fallback  # for good: every vector must come from one model
            if self._vectors is not None:
                self._vectors = self.embedder.embed(self._vector_texts)
            return fn(self.embedder)

    def similar(self, text: str, k: int = 3) -> list[Hit]:
        """Closest documents by cosine similarity (needs an embedder)."""
        if self.embedder is None:
            raise ValueError("similar() needs an embedder")
        [query] = self._with_fallback(lambda e: e.embed([text]))
        scored = ((cosine(query, vector), i) for i, vector in enumerate(self._vectors))
        return [Hit(self._docs[i][0], score) for score, i in _top(scored, k)]

    def search(self, query: str, k: int = 5) -> list[Hit]:
        lexical = self.bm25.search(query, k * 4)
        if self.embedder is None:
            return lexical[:k]
        semantic = [h for h in self.similar(query, k * 4) if h.score >= self.min_similarity]
        fused: dict[str, float] = defaultdict(float)
        for ranking in (lexical, semantic):
            for rank, hit in enumerate(ranking, start=1):
                fused[hit.id] += 1 / (self._rrf_k + rank)
        order = {doc_id: i for i, (doc_id, _) in enumerate(self._docs)}
        best = sorted(fused.items(), key=lambda item: (-item[1], order[item[0]]))[:k]
        return [Hit(doc_id, score) for doc_id, score in best]

    def text(self, doc_id: str) -> str:
        return self._texts[doc_id]
