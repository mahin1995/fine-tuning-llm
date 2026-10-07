"""Training-data access for the ops assistant.

Reuses the project's existing rules (finetune.data): the chat format validation and
question normalisation used for duplicate / leakage checks stay in one place.
Example ids are stable content hashes of the normalised question, so they survive
edits elsewhere in the file and a made-up id simply doesn't resolve.
"""
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from finetune.data.io import load_conversations
from finetune.data.schema import validate_messages
from finetune.data.transforms import question_of


@dataclass(frozen=True)
class Example:
    example_id: str
    question: str
    answer: str


def normalise_question(question: str) -> str:
    """Same normalisation as the dataset validator's duplicate / leakage checks."""
    return question_of([{"role": "user", "content": question}, {"role": "assistant", "content": ""}])


def example_id_for(question: str) -> str:
    return hashlib.sha256(normalise_question(question).encode("utf-8")).hexdigest()[:12]


def _to_example(messages) -> Example:
    question = messages[-2]["content"]
    return Example(example_id_for(question), question, messages[-1]["content"])


class DatasetRepository:
    """Read and write access to the training JSONL. Crew tools only get the read methods."""

    def __init__(self, train_path: Path, eval_path: Path):
        self.train_path = Path(train_path)
        self.eval_path = Path(eval_path)

    # ---- reads
    def examples(self) -> list[Example]:
        if not self.train_path.exists():
            return []
        return [_to_example(m) for m in load_conversations(self.train_path)]

    def get(self, example_id: str) -> Example | None:
        return next((e for e in self.examples() if e.example_id == example_id), None)

    def find_duplicate(self, question: str) -> Example | None:
        key = normalise_question(question)
        return next((e for e in self.examples() if normalise_question(e.question) == key), None)

    def overlaps_eval_set(self, question: str) -> bool:
        if not self.eval_path.exists():
            return False
        key = normalise_question(question)
        return any(question_of(m) == key for m in load_conversations(self.eval_path))

    def search(self, query: str, limit: int = 5) -> list[Example]:
        terms = {t for t in query.lower().split() if len(t) > 2}
        scored = []
        for i, example in enumerate(self.examples()):
            text = f"{example.question} {example.answer}".lower()
            score = sum(term in text for term in terms)
            if score:
                scored.append((-score, i, example))
        return [e for *_, e in sorted(scored)[:limit]]

    def stats(self) -> dict:
        examples = self.examples()
        answer_lengths = [len(e.answer) for e in examples]
        return {
            "examples": len(examples),
            "avg_answer_chars": round(sum(answer_lengths) / len(examples)) if examples else 0,
            "max_answer_chars": max(answer_lengths, default=0),
        }

    # ---- writes (used only by ActionExecutor)
    def append(self, question: str, answer: str) -> Example:
        messages = [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        validate_messages(messages, "new example")
        with self.train_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
        return _to_example(messages)

    def remove(self, example_id: str) -> Example:
        lines = self.train_path.read_text(encoding="utf-8").splitlines()
        kept, removed = [], None
        for line in lines:
            if line.strip() and removed is None:
                example = _to_example(json.loads(line)["messages"])
                if example.example_id == example_id:
                    removed = example
                    continue
            kept.append(line)
        if removed is None:
            raise KeyError(example_id)
        _atomic_write(self.train_path, "\n".join(kept) + ("\n" if kept else ""))
        return removed


def _atomic_write(path: Path, text: str) -> None:
    """Write via a temp file + rename so a crash never leaves a half-written dataset."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
