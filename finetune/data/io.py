"""Load chat datasets from JSONL files."""
import json
from pathlib import Path

from finetune.data.schema import DataError, validate_messages


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
