"""Structured audit log: one JSON object per step, always carrying the correlation id."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol, TextIO


class AuditSink(Protocol):
    def write(self, record: dict) -> None: ...


class MemorySink:
    def __init__(self):
        self.records: list[dict] = []

    def write(self, record):
        self.records.append(record)


class JsonlSink:
    def __init__(self, path: Path):
        self.path = Path(path)

    def write(self, record):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


class StreamSink:
    def __init__(self, stream: TextIO = sys.stderr):
        self.stream = stream

    def write(self, record):
        self.stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


class AuditLog:
    def __init__(self, sinks: list[AuditSink], clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)):
        self._sinks = sinks
        self._clock = clock

    def record(self, correlation_id: str, step: str, **fields) -> dict:
        record = {"ts": self._clock().isoformat(), "correlation_id": correlation_id, "step": step, **fields}
        for sink in self._sinks:
            sink.write(record)
        return record
