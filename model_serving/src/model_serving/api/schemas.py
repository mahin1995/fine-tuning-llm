"""The subset of the OpenAI Chat Completions API this server implements.

Unknown request fields (seed, logprobs, user, ...) are ignored, as most OpenAI-compatible
servers do, so standard clients work unchanged.
"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ToolCallFunction(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: ToolCallFunction


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[dict] | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    messages: list[ChatMessage] = Field(min_length=1, max_length=500)
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    stop: str | list[str] | None = None
    n: int = 1
    stream: bool = False
    stream_options: dict | None = None
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None
    chat_template_kwargs: dict | None = None  # vLLM-compatible extension

    def stop_list(self) -> list[str]:
        if self.stop is None:
            return []
        return [self.stop] if isinstance(self.stop, str) else [s for s in self.stop if s]


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str | None
    tool_calls: list[ToolCall] | None = None


class Choice(BaseModel):
    index: int = 0
    message: AssistantMessage
    finish_reason: Literal["stop", "length", "tool_calls"]


class ChatCompletion(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage
