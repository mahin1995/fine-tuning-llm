"""InferenceEngine port and the plain data it exchanges with the API layer."""
from dataclasses import dataclass, field
from typing import Iterator, Protocol


@dataclass(frozen=True)
class GenerationRequest:
    messages: list[dict]                 # chat-template-ready messages
    max_tokens: int                      # already capped by the server's limit
    temperature: float = 1.0
    top_p: float = 1.0
    stop: list[str] = field(default_factory=list)
    tools: list[dict] | None = None      # OpenAI function schemas, rendered by the chat template
    chat_template_kwargs: dict = field(default_factory=dict)  # merged: model defaults + request


@dataclass(frozen=True)
class Generation:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str                   # "stop" | "length"


@dataclass(frozen=True)
class Delta:
    """One streamed piece. The last one has a finish_reason and the token counts."""
    text: str
    finish_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


class InferenceEngine(Protocol):
    model_name: str
    default_chat_template_kwargs: dict

    def generate(self, req: GenerationRequest) -> Generation: ...

    def stream(self, req: GenerationRequest) -> Iterator[Delta]: ...


def cut_at_stop(text: str, stop: list[str]) -> tuple[str, bool]:
    """Cut `text` at the first stop string (excluded, as in the OpenAI API)."""
    hits = [text.index(s) for s in stop if s and s in text]
    return (text[: min(hits)], True) if hits else (text, False)
