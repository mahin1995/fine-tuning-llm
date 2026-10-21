"""Agent loop behaviour with a scripted backend: no model, no torch."""
import json

import pytest

from agent import Agent, ToolRegistry, make_tool
from agent.builtin_tools import calculator


class ScriptedBackend:
    """Returns pre-written replies in order and records what the agent sent."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, messages, tools):
        self.calls.append((json.loads(json.dumps(messages)), tools))  # snapshot
        return self.replies.pop(0)


def call(name, **arguments):
    return f'<tool_call>\n{json.dumps({"name": name, "arguments": arguments})}\n</tool_call>'


@pytest.fixture
def tools():
    def boom() -> str:
        """Always fails."""
        raise RuntimeError("database down")

    def big() -> str:
        """Returns a lot of text."""
        return "x" * 5000

    return ToolRegistry([make_tool(calculator), make_tool(boom), make_tool(big)])


def test_direct_answer_without_tools(tools):
    backend = ScriptedBackend("It is a stereotype annotation.")
    result = Agent(backend, tools, system_prompt="be brief").run("What is @Service?")
    assert result.stop_reason == "answer"
    assert result.answer == "It is a stereotype annotation."
    assert result.steps == []
    messages, schemas = backend.calls[0]
    assert messages == [{"role": "system", "content": "be brief"}, {"role": "user", "content": "What is @Service?"}]
    assert [s["function"]["name"] for s in schemas] == ["calculator", "boom", "big"]


def test_tool_call_then_answer(tools):
    backend = ScriptedBackend("Computing." + call("calculator", expression="17 * 23"), "17 * 23 = 391")
    seen = []
    result = Agent(backend, tools, on_step=seen.append).run("What is 17*23?")

    assert result.answer == "17 * 23 = 391"
    assert [(s.call.name, s.output, s.ok) for s in result.steps] == [("calculator", "391", True)]
    assert seen == result.steps
    # The second model call sees its own structured tool call and the tool result.
    second_messages = backend.calls[1][0]
    assert second_messages[-2] == {
        "role": "assistant", "content": "Computing.",
        "tool_calls": [{"type": "function", "function": {"name": "calculator", "arguments": {"expression": "17 * 23"}}}],
    }
    assert second_messages[-1] == {"role": "tool", "name": "calculator", "content": "391"}


def test_multiple_calls_in_one_reply(tools):
    backend = ScriptedBackend(call("calculator", expression="1+1") + call("calculator", expression="2+2"), "2 and 4")
    result = Agent(backend, tools).run("q")
    assert [s.output for s in result.steps] == ["2", "4"]
    assert [m["role"] for m in result.messages] == ["user", "assistant", "tool", "tool", "assistant"]


@pytest.mark.parametrize("reply, expected_error", [
    (call("nope"), "unknown tool 'nope'"),
    (call("calculator"), "missing required argument"),
    (call("calculator", expression="import os"), "invalid expression"),
    (call("boom"), "tool boom failed: RuntimeError: database down"),
    ("<tool_call>{broken</tool_call>", "not valid JSON"),
])
def test_errors_are_reported_to_the_model_not_raised(tools, reply, expected_error):
    backend = ScriptedBackend(reply, "sorry, I could not do that")
    result = Agent(backend, tools).run("q")
    assert result.answer == "sorry, I could not do that"
    assert not result.steps[0].ok
    assert expected_error in result.steps[0].output
    assert backend.calls[1][0][-1]["content"].startswith("error:")


def test_malformed_call_keeps_raw_text_for_the_model(tools):
    backend = ScriptedBackend("<tool_call>{broken</tool_call>", "ok")
    Agent(backend, tools).run("q")
    assistant = backend.calls[1][0][-2]
    assert assistant["role"] == "assistant" and "tool_calls" not in assistant


def test_step_limit(tools):
    backend = ScriptedBackend(*[call("calculator", expression=f"{i}+1") for i in range(10)])
    result = Agent(backend, tools, max_steps=3).run("loop forever")
    assert result.stop_reason == "max_steps"
    assert result.answer is None
    assert len(backend.calls) == 3


def test_repeated_identical_call_is_not_rerun(tools):
    counter = {"n": 0}

    def count() -> int:
        """Counts invocations."""
        counter["n"] += 1
        return counter["n"]

    registry = ToolRegistry([make_tool(count)])
    backend = ScriptedBackend(call("count"), call("count"), "done")
    result = Agent(backend, registry).run("q")
    assert counter["n"] == 1
    assert result.steps[1].output == "(repeated call, same result as before) 1"


def test_long_tool_output_is_truncated(tools):
    backend = ScriptedBackend(call("big"), "done")
    result = Agent(backend, tools, max_tool_output_chars=100).run("q")
    assert result.steps[0].output.startswith("x" * 100)
    assert result.steps[0].output.endswith("[truncated 4900 chars]")


def test_history_continues_a_conversation(tools):
    backend = ScriptedBackend("first answer", "second answer")
    agent = Agent(backend, tools, system_prompt="sys")
    first = agent.run("q1")
    second = agent.run("q2", history=first.messages)
    assert [m["content"] for m in second.messages] == ["sys", "q1", "first answer", "q2", "second answer"]
    assert first.messages == [m for m in second.messages[:3]]  # history list is not mutated


def test_invalid_max_steps(tools):
    with pytest.raises(ValueError):
        Agent(ScriptedBackend(), tools, max_steps=0)
