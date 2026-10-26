"""Episodic memory: what happened, when, with what outcome, in SQLite (standard library).

An Episode is a small record written by code: `kind` says what it is ("request", "decline",
"fact", ...), `key` allows exact lookups (an action's idempotency key, a fact's hash), `text`
is searchable text (untrusted when it came from a user), `data` holds structured details.

SqliteEpisodeStore opens a connection per operation: callers may run in different threads
(CrewAI runs flow steps in a thread pool) and sqlite3 connections must not be shared across
threads. Timestamps are stored as UTC ISO 8601 strings, so text order is time order.
"""
import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id      TEXT PRIMARY KEY,
    kind    TEXT NOT NULL,
    ts      TEXT NOT NULL,
    key     TEXT NOT NULL DEFAULT '',
    text    TEXT NOT NULL DEFAULT '',
    outcome TEXT NOT NULL DEFAULT '',
    data    TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS episodes_kind_ts ON episodes (kind, ts);
CREATE INDEX IF NOT EXISTS episodes_kind_key ON episodes (kind, key);
"""


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(ts: datetime) -> str:
    if ts.tzinfo is None:
        raise ValueError("episode timestamps must be timezone-aware")
    return ts.astimezone(timezone.utc).isoformat()


@dataclass(frozen=True)
class Episode:
    id: str
    kind: str
    ts: datetime
    key: str = ""
    text: str = ""
    outcome: str = ""
    data: dict = field(default_factory=dict)


class SqliteEpisodeStore:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as con, con:
            con.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10)

    @staticmethod
    def _episode(row) -> Episode:
        id_, kind, ts, key, text, outcome, data = row
        return Episode(id_, kind, datetime.fromisoformat(ts), key, text, outcome, json.loads(data))

    def add(self, episode: Episode) -> None:
        """Insert or replace (by id)."""
        with closing(self._connect()) as con, con:
            con.execute("INSERT OR REPLACE INTO episodes VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (episode.id, episode.kind, _iso(episode.ts), episode.key, episode.text, episode.outcome,
                         json.dumps(episode.data, ensure_ascii=False, default=str)))

    def get(self, episode_id: str) -> Episode | None:
        with closing(self._connect()) as con:
            row = con.execute("SELECT * FROM episodes WHERE id = ?", (episode_id,)).fetchone()
        return self._episode(row) if row else None

    def latest(self, kind: str, key: str) -> Episode | None:
        with closing(self._connect()) as con:
            row = con.execute("SELECT * FROM episodes WHERE kind = ? AND key = ? ORDER BY ts DESC LIMIT 1",
                              (kind, key)).fetchone()
        return self._episode(row) if row else None

    def find(self, kind: str | None = None, *, since: datetime | None = None, outcome: str | None = None,
             limit: int = 1000) -> list[Episode]:
        """Newest first."""
        where, args = [], []
        for column, value in (("kind", kind), ("outcome", outcome)):
            if value is not None:
                where.append(f"{column} = ?")
                args.append(value)
        if since is not None:
            where.append("ts >= ?")
            args.append(_iso(since))
        sql = "SELECT * FROM episodes" + (" WHERE " + " AND ".join(where) if where else "")
        with closing(self._connect()) as con:
            rows = con.execute(sql + " ORDER BY ts DESC, id LIMIT ?", (*args, limit)).fetchall()
        return [self._episode(r) for r in rows]

    def count(self, kind: str | None = None) -> int:
        with closing(self._connect()) as con:
            if kind is None:
                return con.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            return con.execute("SELECT COUNT(*) FROM episodes WHERE kind = ?", (kind,)).fetchone()[0]

    def prune(self, kind: str, *, older_than: datetime | None = None, keep_latest: int | None = None) -> int:
        """Delete episodes of `kind` older than a time and/or beyond the newest `keep_latest`.
        Returns how many were deleted."""
        deleted = 0
        with closing(self._connect()) as con, con:
            if older_than is not None:
                deleted += con.execute("DELETE FROM episodes WHERE kind = ? AND ts < ?",
                                       (kind, _iso(older_than))).rowcount
            if keep_latest is not None:
                deleted += con.execute(
                    "DELETE FROM episodes WHERE kind = ? AND id NOT IN "
                    "(SELECT id FROM episodes WHERE kind = ? ORDER BY ts DESC, id LIMIT ?)",
                    (kind, kind, keep_latest)).rowcount
        return deleted
