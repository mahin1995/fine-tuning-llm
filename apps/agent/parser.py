"""Extract tool calls from model output (Qwen / Hermes format).

    some optional text
    <tool_call>
    {"name": "calculator", "arguments": {"expression": "2+2"}}
    </tool_call>

Bare JSON without the tags is deliberately NOT treated as a tool call: answers in
this domain often contain JSON examples, and misfiring a tool on them would be worse.
"""
import json
import re
from dataclasses import dataclass, field

# A missing closing tag is tolerated: small models often stop right after the JSON.
_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.DOTALL)


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict

    def signature(self):
        """Stable identity used to detect repeated identical calls."""
        return json.dumps([self.name, self.arguments], sort_keys=True)


@dataclass
class ParsedReply:
    content: str                                     # text outside the tool-call tags
    calls: list[ToolCall] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)  # malformed tool calls, reported back to the model

    @property
    def is_final(self):
        return not self.calls and not self.errors


def _to_call(raw):
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"tool call is not valid JSON ({e.msg})") from e
    if not isinstance(obj, dict) or not isinstance(obj.get("name"), str) or not obj["name"]:
        raise ValueError('tool call must be an object with a "name" string')
    arguments = obj.get("arguments", {})
    if isinstance(arguments, str):  # some models emit arguments as a JSON string
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as e:
            raise ValueError(f'"arguments" string is not valid JSON ({e.msg})') from e
    if not isinstance(arguments, dict):
        raise ValueError('"arguments" must be a JSON object')
    return ToolCall(obj["name"], arguments)


def parse_reply(text: str) -> ParsedReply:
    calls, errors = [], []
    for match in _TOOL_CALL.finditer(text):
        try:
            calls.append(_to_call(match.group(1)))
        except ValueError as e:
            errors.append(str(e))
    content = _TOOL_CALL.sub("", text).strip()
    return ParsedReply(content=content, calls=calls, errors=errors)
