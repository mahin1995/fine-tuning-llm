"""Shared working state for one request: what the dataset tools actually returned.

Agents pass findings to each other as LLM-written text, and the researcher can claim an example
id it never saw. The tools record what they really returned (crew/tools.py: ToolCallRecord.facts);
this module folds those records into one view that later agents (the `verified_facts` tool) and
the deterministic checks (correction.py) rely on. Provenance: code, not an agent's claim.
"""
from dataclasses import dataclass
from typing import Iterable

from ops_crew.schemas import ToolCallRecord


@dataclass(frozen=True)
class VerifiedFacts:
    example_ids: frozenset[str] = frozenset()  # ids some tool returned in this request
    duplicate_of: str | None = None            # find_duplicate's answer, if it was asked
    eval_overlap: bool | None = None           # check_eval_overlap's answer, if it was asked
    similar: tuple[tuple[str, float], ...] = ()  # find_similar: (example_id, score), best first
    eval_similarity: float | None = None

    @classmethod
    def from_tool_calls(cls, calls: Iterable[ToolCallRecord]) -> "VerifiedFacts":
        ids: set[str] = set()
        duplicate_of = eval_overlap = eval_similarity = None
        similar: list[tuple[str, float]] = []
        for call in calls:
            facts = call.facts
            ids.update(facts.get("example_ids", ()))
            if "duplicate_of" in facts:
                duplicate_of = facts["duplicate_of"]
            if "eval_overlap" in facts:
                eval_overlap = facts["eval_overlap"]
            if "eval_similarity" in facts:
                eval_similarity = facts["eval_similarity"]
            similar.extend((s["example_id"], s["score"]) for s in facts.get("similar", ()))
        best = {}
        for example_id, score in similar:
            best[example_id] = max(score, best.get(example_id, score))
        ranked = tuple(sorted(best.items(), key=lambda item: -item[1]))
        return cls(frozenset(ids), duplicate_of, eval_overlap, ranked, eval_similarity)

    def as_dict(self) -> dict:
        return {
            "example_ids": sorted(self.example_ids),
            "duplicate_of": self.duplicate_of,
            "eval_overlap": self.eval_overlap,
            "similar": [{"example_id": i, "score": round(s, 3)} for i, s in self.similar],
            "eval_similarity": None if self.eval_similarity is None else round(self.eval_similarity, 3),
        }
