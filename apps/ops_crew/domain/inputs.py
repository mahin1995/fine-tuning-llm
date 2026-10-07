"""Request intake rules: size limits, cleanup and prompt-injection markers.

Injection markers never block a request by themselves (false positives are easy);
they make the policy demand human approval for any side-effecting action, and they
are recorded in the audit log.
"""
import re
import unicodedata
from dataclasses import dataclass

_INJECTION_PATTERNS = [
    r"ignore (all |any )?(the )?(previous|prior|above) (instructions|prompts?)",
    r"disregard (the |all )?(previous|prior|above|system)",
    r"you are now",
    r"system prompt",
    r"</?\s*(user_request|tool_data|system)\s*>",
    r"\bact as\b",
    r"(new|override|updated) instructions",
    r"(delete|remove|wipe|drop) (all|every|the entire)",
]
_INJECTION = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


class InputRejected(ValueError):
    pass


@dataclass(frozen=True)
class CleanRequest:
    text: str
    suspicious: bool
    markers: tuple[str, ...]


def clean_request(raw: str, max_chars: int) -> CleanRequest:
    if not isinstance(raw, str):
        raise InputRejected("request must be text")
    # Drop control / format characters (incl. zero-width chars used to hide instructions).
    text = "".join(ch for ch in raw if ch in "\n\t" or unicodedata.category(ch)[0] != "C").strip()
    if not text:
        raise InputRejected("request is empty")
    if len(text) > max_chars:
        raise InputRejected(f"request longer than {max_chars} characters")
    markers = tuple(sorted({m.group(0).lower() for m in _INJECTION.finditer(text)}))
    return CleanRequest(text=text, suspicious=bool(markers), markers=markers)
