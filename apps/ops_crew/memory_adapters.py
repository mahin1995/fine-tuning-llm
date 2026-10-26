"""Adapter: ops_crew's memory ports implemented with the generic `memory` package.

The only ops_crew module that imports `memory` (enforced by tests/test_architecture.py); the
domain and the flow see the ports in domain/ports.py. Configuration: config/memory.yaml.

    DatasetSimilarity    SimilarityIndex over the training and eval questions
    SqliteEpisodeMemory  EpisodeMemory: finished requests and human declines (state_dir/memory.db)
    cluster_requests     group similar request texts (consolidation)
"""
import os
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from memory import (
    CachedEmbedder,
    Embedder,
    EmbeddingError,
    Episode,
    HashingEmbedder,
    HttpEmbedder,
    HybridRetriever,
    SqliteEpisodeStore,
    cluster,
)
from ops_crew.crew.config import ConfigError, expand_env
from ops_crew.domain.ports import RequestRecord, SimilarExample, SimilarityReport
from ops_crew.domain.repository import DatasetRepository


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EmbeddingsConfig(_Strict):
    provider: Literal["hashing", "openai_compatible"]
    model: str
    base_url: str
    api_key_env: str | None = None
    timeout: float = Field(default=30, gt=0)


class Thresholds(_Strict):
    near_duplicate: float = Field(ge=0, le=1)
    near_eval: float = Field(ge=0, le=1)
    cluster: float = Field(default=0.5, ge=0, le=1)  # consolidation: group failed requests this similar


class EpisodesConfig(_Strict):
    retention_days: dict[Literal["request", "decline"], int]


class MemoryConfig(_Strict):
    embeddings: EmbeddingsConfig
    similarity: dict[Literal["hashing", "openai_compatible"], Thresholds]
    episodes: EpisodesConfig


def load_memory_config(config_dir: Path, env: dict | None = None) -> MemoryConfig:
    env = dict(os.environ) if env is None else env
    path = Path(config_dir) / "memory.yaml"
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        data["embeddings"] = {k: expand_env(v, env) if isinstance(v, str) else v
                              for k, v in data["embeddings"].items()}
        config = MemoryConfig.model_validate(data)
    except (OSError, KeyError, TypeError, yaml.YAMLError, ValueError) as e:
        raise ConfigError(f"invalid memory config in {path}: {e}") from e
    if {"hashing", config.embeddings.provider} - set(config.similarity):
        raise ConfigError(f"{path}: similarity thresholds needed for 'hashing' and '{config.embeddings.provider}'")
    return config


def build_embedder(config: EmbeddingsConfig, env: dict | None = None) -> Embedder:
    if config.provider == "hashing":
        return HashingEmbedder()
    env = dict(os.environ) if env is None else env
    api_key = env.get(config.api_key_env) if config.api_key_env else None
    return CachedEmbedder(HttpEmbedder(config.base_url, config.model, api_key=api_key, timeout=config.timeout))


class DatasetSimilarity:
    """SimilarityIndex over the training and eval questions (memory.HybridRetriever).

    Indexes are rebuilt when a dataset file changes (size + modification time); a CachedEmbedder
    makes a rebuild embed only new questions. If the embedding provider fails, the hashing
    embedder takes over for good and its thresholds apply; `errors` says what failed.
    Search leaves out semantic matches below half the near-duplicate threshold."""

    def __init__(self, repo: DatasetRepository, embedder: Embedder, thresholds: dict[str, Thresholds]):
        self._repo = repo
        self._embedder = embedder
        self._fallback = HashingEmbedder()
        self._thresholds = thresholds
        self._lock = threading.Lock()
        self._indexes: dict[str, tuple[tuple, HybridRetriever]] = {}

    @classmethod
    def from_config(cls, repo: DatasetRepository, config: MemoryConfig, env: dict | None = None):
        return cls(repo, build_embedder(config.embeddings, env), config.similarity)

    @property
    def errors(self) -> list[str]:
        """Embedding provider failures that made an index switch to the hashing fallback."""
        return [e for _, retriever in self._indexes.values() for e in retriever.errors]

    @staticmethod
    def _signature(path: Path) -> tuple:
        try:
            stat = Path(path).stat()
        except FileNotFoundError:
            return (0, 0)
        return (stat.st_size, stat.st_mtime_ns)

    def _thresholds_for(self, embedder: Embedder) -> Thresholds:
        return self._thresholds["hashing" if isinstance(embedder, HashingEmbedder) else "openai_compatible"]

    def _index(self, name: str) -> HybridRetriever:
        path = self._repo.train_path if name == "train" else self._repo.eval_path
        signature = self._signature(path)
        with self._lock:
            cached = self._indexes.get(name)
            if cached is not None and cached[0] == signature:
                return cached[1]
            if name == "train":
                examples = self._repo.examples()
                docs = [(e.example_id, f"{e.question}\n{e.question}\n{e.answer}") for e in examples]
                questions = [e.question for e in examples]
            else:
                questions = self._repo.eval_questions()
                docs = [(f"eval-{i}", q) for i, q in enumerate(questions)]  # ids never leave this class
            retriever = HybridRetriever(docs, self._embedder, fallback=self._fallback, vector_texts=questions,
                                        min_similarity=self._thresholds_for(self._embedder).near_duplicate / 2)
            self._indexes[name] = (signature, retriever)
            return retriever

    def search(self, query: str, k: int) -> list[str]:
        return [hit.id for hit in self._index("train").search(query, k)]

    def similar_examples(self, text: str, k: int = 3) -> list[SimilarExample]:
        return [SimilarExample(hit.id, hit.score) for hit in self._index("train").similar(text, k)]

    def report(self, text: str) -> SimilarityReport:
        train, evals = self._index("train"), self._index("eval")
        closest, eval_hits = train.similar(text, 1), evals.similar(text, 1)
        return SimilarityReport(
            closest_example=SimilarExample(closest[0].id, closest[0].score) if closest else None,
            eval_score=eval_hits[0].score if eval_hits else 0.0,
            near_duplicate_threshold=self._thresholds_for(train.embedder).near_duplicate,
            near_eval_threshold=self._thresholds_for(evals.embedder).near_eval,
            embedder=train.embedder.name,
        )


# ------------------------------------------------------------------ episodes

REQUEST, DECLINE = "request", "decline"


class SqliteEpisodeMemory:
    """EpisodeMemory on memory.SqliteEpisodeStore. Requests are keyed by correlation id, declines
    by the declined action's idempotency key (without request id), so the same action is
    recognised in any later request."""

    def __init__(self, path: Path):
        self.store = SqliteEpisodeStore(path)

    def record_request(self, record: RequestRecord) -> None:
        self.store.add(Episode(
            id=record.correlation_id, kind=REQUEST, ts=record.ts, key=record.reply_to or "", text=record.request,
            outcome=record.outcome,
            data={"role": record.role, "message": record.message, "intent": record.intent,
                  "plan_intents": list(record.plan_intents), "step_outcomes": list(record.step_outcomes),
                  "reflection_score": record.reflection_score, "reply_to": record.reply_to},
        ))

    @staticmethod
    def _record(episode: Episode) -> RequestRecord:
        d = episode.data
        return RequestRecord(
            correlation_id=episode.id, ts=episode.ts, role=d["role"], request=episode.text, outcome=episode.outcome,
            message=d["message"], intent=d["intent"], plan_intents=tuple(d["plan_intents"]),
            step_outcomes=tuple(d["step_outcomes"]), reflection_score=d["reflection_score"], reply_to=d["reply_to"],
        )

    def get_request(self, correlation_id: str) -> RequestRecord | None:
        episode = self.store.get(correlation_id)
        return self._record(episode) if episode is not None and episode.kind == REQUEST else None

    def requests(self, since: datetime | None = None) -> list[RequestRecord]:
        """Newest first."""
        return [self._record(e) for e in self.store.find(REQUEST, since=since, limit=1_000_000)]

    def record_decline(self, action_key: str, correlation_id: str, ts: datetime) -> None:
        self.store.add(Episode(id=f"{correlation_id}:{action_key}", kind=DECLINE, ts=ts, key=action_key))

    def declined_before(self, action_key: str) -> datetime | None:
        episode = self.store.latest(DECLINE, action_key)
        return episode.ts if episode is not None else None

    def prune(self, now: datetime, retention_days: dict[str, int]) -> dict[str, int]:
        """Forget episodes past their retention; declines usually live longer than requests."""
        return {kind: self.store.prune(kind, older_than=now - timedelta(days=days))
                for kind, days in retention_days.items()}


def cluster_requests(texts: list[str], config: MemoryConfig, env: dict | None = None) -> list[list[int]]:
    """Group similar request texts with the configured embedder (hashing if it is unavailable)."""
    embedder = build_embedder(config.embeddings, env)
    try:
        return cluster(texts, embedder, config.similarity[config.embeddings.provider].cluster)
    except EmbeddingError:
        return cluster(texts, HashingEmbedder(), config.similarity["hashing"].cluster)
