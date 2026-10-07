import pytest

from agent.parser import ToolCall, parse_reply


def test_plain_text_is_final_answer():
    parsed = parse_reply("REQUIRED joins the existing transaction.")
    assert parsed.is_final
    assert parsed.content == "REQUIRED joins the existing transaction."


def test_single_tool_call_with_surrounding_text():
    parsed = parse_reply('Let me compute.\n<tool_call>\n{"name": "calculator", "arguments": {"expression": "2+2"}}\n'
                         '</tool_call>')
    assert parsed.calls == [ToolCall("calculator", {"expression": "2+2"})]
    assert parsed.content == "Let me compute."
    assert not parsed.is_final


def test_multiple_calls_and_missing_closing_tag():
    parsed = parse_reply('<tool_call>{"name": "a", "arguments": {}}</tool_call>'
                         '<tool_call>{"name": "b", "arguments": {"x": 1}}')
    assert [c.name for c in parsed.calls] == ["a", "b"]
    assert parsed.calls[1].arguments == {"x": 1}


def test_arguments_as_json_string_and_missing_arguments():
    parsed = parse_reply('<tool_call>{"name": "a", "arguments": "{\\"q\\": \\"x\\"}"}</tool_call>'
                         '<tool_call>{"name": "b"}</tool_call>')
    assert parsed.calls == [ToolCall("a", {"q": "x"}), ToolCall("b", {})]


@pytest.mark.parametrize("raw, error", [
    ("{not json}", "not valid JSON"),
    ('{"arguments": {}}', '"name"'),
    ('{"name": "a", "arguments": [1]}', "JSON object"),
    ('{"name": "a", "arguments": "{bad"}', "arguments"),
    ('["a"]', '"name"'),
])
def test_malformed_calls_become_errors(raw, error):
    parsed = parse_reply(f"<tool_call>{raw}</tool_call>")
    assert parsed.calls == []
    assert len(parsed.errors) == 1 and error in parsed.errors[0]
    assert not parsed.is_final


def test_bare_json_is_not_a_tool_call():
    # Answers in this domain contain JSON examples; only tagged calls count.
    parsed = parse_reply('Return {"name": "calculator", "arguments": {}} from the controller.')
    assert parsed.is_final


def test_signature_ignores_key_order():
    assert ToolCall("a", {"x": 1, "y": 2}).signature() == ToolCall("a", {"y": 2, "x": 1}).signature()
