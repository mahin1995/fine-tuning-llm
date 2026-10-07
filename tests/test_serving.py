import pytest
from fastapi.testclient import TestClient

from finetune.serving.app import create_app


class FakeChat:
    name = "fake-model"

    def __init__(self):
        self.calls = []

    def generate(self, messages, params):
        self.calls.append((messages, params))
        return f"echo: {messages[-1]['content']}"


@pytest.fixture
def client_and_chat():
    chat = FakeChat()
    with TestClient(create_app(lambda: chat)) as client:
        yield client, chat


def test_index_and_health(client_and_chat):
    client, _ = client_and_chat
    assert "<title>Qwen Chat</title>" in client.get("/").text
    assert client.get("/health").json() == {"status": "ok", "model": "fake-model"}


def test_chat_returns_reply_and_passes_params(client_and_chat):
    client, chat = client_and_chat
    res = client.post("/chat", json={
        "messages": [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}],
        "temperature": 0,
        "max_new_tokens": 16,
    })
    assert res.status_code == 200
    assert res.json() == {"reply": "echo: hi"}
    messages, params = chat.calls[0]
    assert messages[0] == {"role": "system", "content": "be brief"}
    assert params.temperature == 0 and params.max_new_tokens == 16


@pytest.mark.parametrize(
    "body",
    [
        {"messages": []},
        {"messages": [{"role": "assistant", "content": "x"}]},
        {"messages": [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]},
        {"messages": [{"role": "user", "content": ""}]},
        {"messages": [{"role": "hacker", "content": "x"}]},
        {"messages": [{"role": "user", "content": "x"}], "max_new_tokens": 100000},
        {"messages": [{"role": "user", "content": "x"}], "temperature": -1},
        {"messages": [{"role": "user", "content": "x" * 8001}]},
    ],
)
def test_chat_rejects_invalid_requests(client_and_chat, body):
    client, chat = client_and_chat
    assert client.post("/chat", json=body).status_code == 422
    assert chat.calls == []
