"""Tool calling for models that emit Hermes-style calls (Qwen, many others):

    <tool_call>
    {"name": "search", "arguments": {"query": "x"}}
    </tool_call>

parse_tool_calls turns that text into OpenAI `tool_calls`; to_template_messages turns OpenAI
messages (arguments as JSON strings) back into what chat templates expect (dicts).
"""
import json
import re
import uuid

_CALL = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.DOTALL)


def parse_tool_calls(text: str) -> tuple[str, list[dict]]:
    """Returns (remaining content, tool_calls). If any block is malformed, the whole output is
    returned as plain content with no calls: the server never guesses a call the model didn't make."""
    calls = []
    for match in _CALL.finditer(text):
        try:
            obj = json.loads(match.group(1))
            name, args = obj["name"], obj.get("arguments", {})
            if not isinstance(name, str) or not name:
                raise ValueError("tool name must be a non-empty string")
        except (ValueError, KeyError, TypeError):
            return text, []
        calls.append({
            "id": f"call_{uuid.uuid4().hex[:12]}",
            "type": "function",
            "function": {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args)},
        })
    if not calls:
        return text, []
    return _CALL.sub("", text).strip(), calls


def to_template_messages(messages: list[dict]) -> list[dict]:
    converted = []
    for message in messages:
        message = dict(message)
        if message.get("content") is None:
            message["content"] = ""
        if message.get("tool_calls"):
            calls = []
            for call in message["tool_calls"]:
                function = dict(call.get("function", {}))
                try:
                    function["arguments"] = json.loads(function.get("arguments") or "{}")
                except (TypeError, ValueError):
                    pass  # keep the raw string; templates accept both
                calls.append({**call, "function": function})
            message["tool_calls"] = calls
        converted.append(message)
    return converted
