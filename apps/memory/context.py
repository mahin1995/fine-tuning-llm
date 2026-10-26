"""Context engineering: count tokens, keep a chat history inside a budget, pack recalled items.

compact_messages keeps a history inside a token budget without an LLM (a summary can drop the
delimiters around untrusted text, or invent facts):

    1. the leading system message(s) always stay
    2. a turn is a user message plus every assistant / tool message after it, so a tool call
       and its result are never separated
    3. over budget: clear tool outputs in older turns first (oldest first), then drop the
       oldest turns; the newest turn always stays, even if it alone is over budget

Cost: O(number of messages) token counts per call.
"""
import json
import math
from dataclasses import dataclass
from typing import Protocol

MESSAGE_OVERHEAD = 4  # role markers and separators a chat template adds per message
CLEARED_TOOL_OUTPUT = "[tool output cleared to save context]"


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class ApproxTokenCounter:
    """About 4 characters per token for English text and code. It errs high for short words,
    which is the safe side for a budget; use the model's tokenizer when it is at hand."""

    def __init__(self, chars_per_token: float = 4.0):
        self.chars_per_token = chars_per_token

    def count(self, text: str) -> int:
        return math.ceil(len(text) / self.chars_per_token) if text else 0


def message_tokens(message: dict, counter: TokenCounter) -> int:
    calls = message.get("tool_calls")
    return (counter.count(message.get("content") or "") + (counter.count(json.dumps(calls)) if calls else 0)
            + MESSAGE_OVERHEAD)


def count_messages(messages: list[dict], counter: TokenCounter) -> int:
    return sum(message_tokens(m, counter) for m in messages)


@dataclass(frozen=True)
class CompactionReport:
    tokens_before: int
    tokens_after: int
    cleared_tool_outputs: int = 0
    dropped_turns: int = 0

    @property
    def changed(self) -> bool:
        return bool(self.cleared_tool_outputs or self.dropped_turns)


def _split(messages: list[dict]) -> tuple[list[dict], list[list[dict]]]:
    """(leading system messages, turns). Messages before the first user message form a turn too."""
    start = 0
    while start < len(messages) and messages[start].get("role") == "system":
        start += 1
    turns: list[list[dict]] = []
    for message in messages[start:]:
        if message.get("role") == "user" or not turns:
            turns.append([])
        turns[-1].append(message)
    return messages[:start], turns


def compact_messages(messages: list[dict], budget: int,
                     counter: TokenCounter | None = None) -> tuple[list[dict], CompactionReport]:
    """Return a copy of `messages` that fits `budget` tokens where possible (see module doc).
    The input list and its dicts are never modified."""
    counter = counter or ApproxTokenCounter()
    before = count_messages(messages, counter)
    if before <= budget:
        return list(messages), CompactionReport(before, before)

    system, turns = _split(messages)
    turns = [list(turn) for turn in turns]
    total, cleared, dropped = before, 0, 0
    for turn in turns[:-1]:  # 1) clear tool outputs, oldest turn first
        for i, message in enumerate(turn):
            if total <= budget:
                break
            if message.get("role") == "tool" and message.get("content") != CLEARED_TOOL_OUTPUT:
                replacement = {**message, "content": CLEARED_TOOL_OUTPUT}
                total += message_tokens(replacement, counter) - message_tokens(message, counter)
                turn[i] = replacement
                cleared += 1
    while total > budget and len(turns) > 1:  # 2) drop the oldest turns
        total -= count_messages(turns.pop(0), counter)
        dropped += 1
    compacted = list(system) + [m for turn in turns for m in turn]
    return compacted, CompactionReport(before, total, cleared, dropped)


def pack(items: list[str], budget: int, counter: TokenCounter | None = None) -> list[str]:
    """Take items in order (most relevant first) while they fit in `budget` tokens; skip the rest."""
    counter = counter or ApproxTokenCounter()
    kept, used = [], 0
    for item in items:
        cost = counter.count(item) + 1
        if used + cost <= budget:
            kept.append(item)
            used += cost
    return kept


def delimit(tag: str, text: str) -> str:
    """Wrap untrusted text as data. Our tag inside the text is neutralised so it can't close the block."""
    text = text.replace(f"</{tag}>", f"<\\/{tag}>").replace(f"<{tag}>", f"<\\{tag}>")
    return f"<{tag}>\n{text}\n</{tag}>"
