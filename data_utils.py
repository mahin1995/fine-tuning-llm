"""Dataset loading, validation and conversion shared by train.py, evaluate.py and validate_data.py.

Data format (one JSON object per line, Qwen chat format):
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

An optional leading "system" message and multi-turn conversations are allowed.
Only the final assistant turn is trained on (prompt-completion format), so the
model learns the answer, not the question.
"""
import json
import random
from pathlib import Path

# Qwen3 renders an empty <think></think> block when thinking is disabled. Training
# and inference must use the same setting, otherwise the prompt format differs.
CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}

VALID_ROLES = {"system", "user", "assistant"}


class DataError(ValueError):
    """Raised when a dataset line does not follow the expected chat format."""


def validate_messages(messages, where="example"):
    if not isinstance(messages, list) or len(messages) < 2:
        raise DataError(f"{where}: 'messages' must be a list with at least a user and an assistant turn")

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise DataError(f"{where}: message {i} is not an object")
        role, content = msg.get("role"), msg.get("content")
        if role not in VALID_ROLES:
            raise DataError(f"{where}: message {i} has invalid role {role!r}")
        if not isinstance(content, str) or not content.strip():
            raise DataError(f"{where}: message {i} has empty or non-string content")
        if role == "system" and i != 0:
            raise DataError(f"{where}: system message is only allowed as the first message")

    turns = messages[1:] if messages[0]["role"] == "system" else messages
    expected = ["user", "assistant"]
    for i, msg in enumerate(turns):
        if msg["role"] != expected[i % 2]:
            raise DataError(f"{where}: roles must alternate user/assistant, got {msg['role']!r} at turn {i}")
    if turns[-1]["role"] != "assistant":
        raise DataError(f"{where}: last message must be from the assistant")


def load_conversations(path):
    """Read and validate a JSONL chat dataset. Blank lines are skipped."""
    path = Path(path)
    conversations = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            where = f"{path}:{line_no}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise DataError(f"{where}: invalid JSON ({e})") from e
            if not isinstance(row, dict) or "messages" not in row:
                raise DataError(f"{where}: expected an object with a 'messages' key")
            validate_messages(row["messages"], where)
            conversations.append(row["messages"])
    if not conversations:
        raise DataError(f"{path}: no examples found")
    return conversations


def to_prompt_completion(messages):
    """Split a conversation into TRL's conversational prompt-completion format."""
    return {
        "prompt": messages[:-1],
        "completion": messages[-1:],
        "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
    }


def train_eval_split(conversations, eval_ratio, seed):
    """Shuffle and split. Returns (train, eval); eval is empty when eval_ratio == 0."""
    if not 0 <= eval_ratio < 1:
        raise ValueError("eval_ratio must be in [0, 1)")
    rows = list(conversations)
    random.Random(seed).shuffle(rows)
    n_eval = round(len(rows) * eval_ratio)
    if eval_ratio > 0 and n_eval == 0:
        n_eval = 1
    if n_eval >= len(rows):
        raise DataError(f"eval split of {n_eval} leaves no training data ({len(rows)} examples)")
    return rows[n_eval:], rows[:n_eval]
