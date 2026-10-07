"""The OpenAI-compatible API with FakeEngine: self-contained, no model, no network."""
import json

import pytest
from fastapi.testclient import TestClient

from model_serving.api.app import create_app
from model_serving.engines.fake import FakeEngine

TOOL_CALL = '<tool_call>\n{"name": "search", "arguments": {"query": "N+1"}}\n</tool_call>'
TOOLS = [{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}]


def client(engine=None, limit=50, api_key=None):
    engine = engine or FakeEngine(model_name="m", default_chat_template_kwargs={"enable_thinking": False})
    return TestClient(create_app(engine, max_tokens_limit=limit, api_key=api_key)), engine


def chat(c, **body):
    payload = {"model": "m", "messages": [{"role": "user", "content": "hello world"}], **body}
    return c.post("/v1/chat/completions", json=payload)


def test_health_and_models():
    c, _ = client()
    assert c.get("/health").json() == {"status": "ok", "model": "m"}
    data = c.get("/v1/models").json()
    assert data["object"] == "list"
    assert data["data"][0]["id"] == "m" and data["data"][0]["max_tokens_limit"] == 50


def test_completion_shape_and_usage():
    c, _ = client()
    r = chat(c, max_tokens=10)
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion" and body["id"].startswith("chatcmpl-")
    assert body["choices"] == [{"index": 0, "message": {"role": "assistant", "content": "echo: hello world"},
                                "finish_reason": "stop"}]
    assert body["usage"] == {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}
    assert "x-max-tokens-limited" not in r.headers


def test_generation_policy_comes_from_the_request():
    c, engine = client()
    chat(c, max_tokens=7, temperature=0.2, top_p=0.9, stop=["END"])
    req = engine.requests[-1]
    assert (req.max_tokens, req.temperature, req.top_p, req.stop) == (7, 0.2, 0.9, ["END"])


def test_max_tokens_is_capped_not_rejected():
    c, engine = client(limit=5)
    r = chat(c, max_tokens=4096)
    assert r.status_code == 200 and r.headers["x-max-tokens-limited"] == "true"
    assert engine.requests[-1].max_tokens == 5
    chat(c)  # omitted -> the limit
    assert engine.requests[-1].max_tokens == 5
    chat(c, max_completion_tokens=3)  # newer OpenAI field name
    assert engine.requests[-1].max_tokens == 3


def test_length_finish_reason():
    c, _ = client(FakeEngine(model_name="m", replies=["one two three four five"]))
    body = chat(c, max_tokens=2).json()
    assert body["choices"][0]["message"]["content"] == "one two"
    assert body["choices"][0]["finish_reason"] == "length"


def test_chat_template_kwargs_defaults_and_override():
    c, engine = client()
    chat(c)
    assert engine.requests[-1].chat_template_kwargs == {"enable_thinking": False}
    chat(c, chat_template_kwargs={"enable_thinking": True, "x": 1})
    assert engine.requests[-1].chat_template_kwargs == {"enable_thinking": True, "x": 1}


def test_message_normalisation():
    c, engine = client()
    chat(c, messages=[
        {"role": "developer", "content": "be brief"},
        {"role": "user", "content": [{"type": "text", "text": "a "}, {"type": "text", "text": "b"}]},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "search", "arguments": "{\"q\": 1}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "result"},
    ])
    messages = engine.requests[-1].messages
    assert messages[0] == {"role": "system", "content": "be brief"}
    assert messages[1] == {"role": "user", "content": "a b"}
    assert messages[2]["tool_calls"][0]["function"]["arguments"] == {"q": 1}  # dict for chat templates
    assert messages[3] == {"role": "tool", "content": "result", "tool_call_id": "c1"}


def test_tool_calls_are_returned_in_openai_format():
    c, engine = client(FakeEngine(model_name="m", replies=[TOOL_CALL]))
    body = chat(c, tools=TOOLS).json()
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    [call] = choice["message"]["tool_calls"]
    assert call["type"] == "function" and call["id"].startswith("call_")
    assert call["function"] == {"name": "search", "arguments": '{"query": "N+1"}'}
    assert "content" not in choice["message"]  # null content omitted
    assert engine.requests[-1].tools == TOOLS


def test_tool_choice_none_hides_tools_and_text_stays_text():
    c, engine = client(FakeEngine(model_name="m", replies=[TOOL_CALL]))
    body = chat(c, tools=TOOLS, tool_choice="none").json()
    assert engine.requests[-1].tools is None
    assert body["choices"][0]["finish_reason"] == "stop"  # not parsed when tools are off


def sse(r):
    events = [line[6:] for line in r.text.splitlines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    return [json.loads(e) for e in events[:-1]]


def test_streaming():
    c, _ = client()
    r = chat(c, stream=True, stream_options={"include_usage": True})
    assert r.headers["content-type"].startswith("text/event-stream")
    chunks = sse(r)
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    text = "".join(ch["choices"][0]["delta"].get("content", "") for ch in chunks if ch["choices"])
    assert text == "echo: hello world"
    assert [ch["choices"][0]["finish_reason"] for ch in chunks if ch["choices"]][-1] == "stop"
    assert chunks[-1]["choices"] == [] and chunks[-1]["usage"]["total_tokens"] == 5


def test_streaming_with_tools_sends_the_call_once():
    c, _ = client(FakeEngine(model_name="m", replies=[TOOL_CALL]))
    chunks = sse(chat(c, stream=True, tools=TOOLS))
    calls = [ch["choices"][0]["delta"]["tool_calls"] for ch in chunks if "tool_calls" in ch["choices"][0]["delta"]]
    assert len(calls) == 1 and calls[0][0]["index"] == 0 and calls[0][0]["function"]["name"] == "search"
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


@pytest.mark.parametrize("body, status, message", [
    ({"model": "other"}, 404, "model 'other' is not served here"),
    ({"n": 2}, 400, "only n=1"),
    ({"tool_choice": "required"}, 400, "tool_choice must be"),
    ({"temperature": 5}, 400, "temperature: Input should be less than or equal to 2"),
    ({"max_tokens": 0}, 400, "max_tokens: Input should be greater than or equal to 1"),
    ({"messages": []}, 400, "messages: List should have at least 1 item"),
    ({"messages": [{"role": "user", "content": [{"type": "image_url"}]}]}, 400, "text only"),
])
def test_errors_use_the_openai_error_format(body, status, message):
    c, _ = client()
    r = chat(c, **body)
    assert r.status_code == status
    assert message in r.json()["error"]["message"]


def test_api_key():
    c, _ = client(api_key="s3cret")
    assert chat(c).status_code == 401
    assert chat(c).json()["error"]["code"] == "invalid_api_key"
    ok = c.post("/v1/chat/completions", headers={"Authorization": "Bearer s3cret"},
                json={"model": "m", "messages": [{"role": "user", "content": "x"}]})
    assert ok.status_code == 200
    assert c.get("/health").status_code == 200  # health stays open for load balancers
