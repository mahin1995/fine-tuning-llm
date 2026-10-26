"""Rules for a valid chat conversation.

Data format (one JSON object per line, Qwen chat format):
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

An optional leading "system" message and multi-turn conversations are allowed.
"""

VALID_ROLES = {"system", "user", "assistant"}


class DataError(ValueError):
    """Raised when a conversation does not follow the expected chat format."""


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


def validate_prompt(messages, where="request"):
    """Validate a conversation that ends with a user turn (an inference prompt)."""
    validate_messages(list(messages) + [{"role": "assistant", "content": "-"}], where)
