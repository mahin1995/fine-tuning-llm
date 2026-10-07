"""The only thing the agent loop needs from a language model."""
from typing import Protocol


class ChatBackend(Protocol):
    def complete(self, messages: list[dict], tools: list[dict]) -> str:
        """Return the model's next reply as raw text.

        `messages` use the chat format (system/user/assistant/tool roles, assistant
        messages may carry `tool_calls`); `tools` are OpenAI-style function schemas.
        Tool calls must appear in the text as <tool_call>{"name": ..., "arguments": {...}}</tool_call>.
        """
        ...
