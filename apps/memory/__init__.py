"""Memory toolkit for agents: retrieval, context budgets and episodic storage.

Generic and domain-agnostic, standard library only (enforced by tests/test_architecture.py):
it knows nothing about datasets, roles or prompts. Callers decide what to remember.

    retrieval      Bm25Index, Embedder port (HashingEmbedder offline, HttpEmbedder for any
                   OpenAI-compatible /embeddings endpoint, CachedEmbedder), HybridRetriever
                   (BM25 + vectors with Reciprocal Rank Fusion, provider fallback)
    context        TokenCounter, compact_messages (keep a chat history inside a token budget
                   without an LLM), pack (fit recalled items into a budget), delimit
    episodes       Episode, SqliteEpisodeStore (what happened, when, with what outcome)
    consolidation  cluster (group similar episodes)
    text           normalize, terms, char_ngrams

Used by apps/agent directly and by ops_crew through ops_crew/memory_adapters.py.
"""
from memory.consolidation import cluster
from memory.context import (
    ApproxTokenCounter,
    CompactionReport,
    TokenCounter,
    compact_messages,
    count_messages,
    delimit,
    pack,
)
from memory.episodes import Episode, SqliteEpisodeStore, utcnow
from memory.retrieval import (
    Bm25Index,
    CachedEmbedder,
    Embedder,
    EmbeddingError,
    HashingEmbedder,
    Hit,
    HttpEmbedder,
    HybridRetriever,
    cosine,
)
from memory.text import normalize, terms

__all__ = [
    "ApproxTokenCounter", "Bm25Index", "CachedEmbedder", "CompactionReport", "Embedder", "EmbeddingError",
    "Episode", "HashingEmbedder", "Hit", "HttpEmbedder", "HybridRetriever", "SqliteEpisodeStore", "TokenCounter",
    "cluster", "compact_messages", "cosine", "count_messages", "delimit", "normalize", "pack", "terms", "utcnow",
]
