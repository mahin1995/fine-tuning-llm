"""The agent loop: ask the model, run the tools it calls, feed results back, repeat.

    for step in range(max_steps):
        reply = backend.complete(messages, tools)
        no tool call in reply  -> that's the final answer
        otherwise              -> run each call, append results as "tool" messages

Safety rails: a step limit, argument validation (ToolRegistry), tool exceptions
and malformed calls are returned to the model as errors instead of crashing,
repeated identical calls are not re-executed, and tool output is truncated.
"""
from dataclasses import dataclass, field
from typing import Callable, Literal

from agent.backend import ChatBackend
from agent.parser import ParsedReply, ToolCall, parse_reply
from agent.tools import ToolError, ToolRegistry


@dataclass
class Step:
    call: ToolCall | None  # None when the model emitted a malformed tool call
    output: str            # tool result or error text, exactly as shown to the model
    ok: bool


@dataclass
class AgentResult:
    answer: str | None                        # None when the step limit was hit
    stop_reason: Literal["answer", "max_steps"]
    steps: list[Step] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)  # full transcript, reusable as history


class Agent:
    def __init__(
        self,
        backend: ChatBackend,
        tools: ToolRegistry,
        *,
        system_prompt: str | None = None,
        max_steps: int = 5,
        max_tool_output_chars: int = 2000,
        on_step: Callable[[Step], None] | None = None,
    ):
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        self.backend = backend
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.max_tool_output_chars = max_tool_output_chars
        self.on_step = on_step

    def run(self, question: str, history: list[dict] | None = None) -> AgentResult:
        """Answer `question`. Pass a previous result's `messages` as `history` to continue a conversation."""
        messages = list(history) if history else self._initial_messages()
        messages.append({"role": "user", "content": question})
        steps: list[Step] = []
        executed: dict[str, str] = {}  # call signature -> output, to avoid re-running identical calls
        schemas = self.tools.schemas()

        for _ in range(self.max_steps):
            parsed = parse_reply(self.backend.complete(messages, schemas))
            if parsed.is_final:
                messages.append({"role": "assistant", "content": parsed.content})
                return AgentResult(parsed.content, "answer", steps, messages)

            messages.append(self._assistant_message(parsed))
            for error in parsed.errors:
                self._record(steps, messages, Step(None, f"error: {error}", ok=False))
            for call in parsed.calls:
                self._record(steps, messages, self._execute(call, executed), name=call.name)

        return AgentResult(None, "max_steps", steps, messages)

    def _initial_messages(self):
        return [{"role": "system", "content": self.system_prompt}] if self.system_prompt else []

    @staticmethod
    def _assistant_message(parsed: ParsedReply):
        message = {"role": "assistant", "content": parsed.content}
        if parsed.calls:
            message["tool_calls"] = [
                {"type": "function", "function": {"name": c.name, "arguments": c.arguments}} for c in parsed.calls
            ]
        return message

    def _execute(self, call: ToolCall, executed: dict) -> Step:
        signature = call.signature()
        if signature in executed:
            return Step(call, f"(repeated call, same result as before) {executed[signature]}", ok=True)
        try:
            output, ok = self._truncate(self.tools.execute(call.name, call.arguments)), True
        except ToolError as e:
            output, ok = f"error: {e}", False
        except Exception as e:  # a buggy tool must not kill the loop; the model can try something else
            output, ok = f"error: tool {call.name} failed: {type(e).__name__}: {e}", False
        if ok:
            executed[signature] = output
        return Step(call, output, ok)

    def _truncate(self, text):
        if len(text) <= self.max_tool_output_chars:
            return text
        dropped = len(text) - self.max_tool_output_chars
        return f"{text[: self.max_tool_output_chars]}\n...[truncated {dropped} chars]"

    def _record(self, steps, messages, step: Step, name=None):
        steps.append(step)
        message = {"role": "tool", "content": step.output}
        if name:
            message["name"] = name
        messages.append(message)
        if self.on_step:
            self.on_step(step)
