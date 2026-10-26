"""Answering a clarifying question in a later request (episodic memory, `--reply-to <id>`).

Without memory every request stands alone: "Remove that example." -> "Which example?" ->
"the one about bean scopes" loses the first half. With a reply, code joins the earlier request,
the question and the answer into one request text. All three parts are treated as untrusted
user text (the question came from the model), so the joined text goes through intake again:
cleanup, size limit, injection markers.
"""
from datetime import datetime, timedelta

from ops_crew.domain.ports import RequestRecord

CLARIFICATION_NEEDED = "clarification_needed"


def reply_problem(record: RequestRecord | None, correlation_id: str, now: datetime, ttl: timedelta) -> str | None:
    """Why this reply can't be accepted, or None. Texts are written here (ids are ours)."""
    if record is None:
        return f"nothing to reply to: no request {correlation_id} in memory"
    if record.outcome != CLARIFICATION_NEEDED:
        return f"request {correlation_id} did not ask a question (its outcome was {record.outcome})"
    if now - record.ts > ttl:
        return f"the question from request {correlation_id} has expired (older than {int(ttl.total_seconds() // 60)} " \
               "minutes); send a new request"
    return None


def followup_text(record: RequestRecord, answer: str) -> str:
    return f"Earlier request: {record.request}\nClarifying question: {record.message}\nAnswer: {answer}"
