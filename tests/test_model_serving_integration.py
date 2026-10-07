"""model_serving against the real pieces: the transformers engine on the tiny Qwen3 model,
the official OpenAI SDK as a client, and consistency with finetune's model profiles."""
import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("torch")

from fastapi.testclient import TestClient  # noqa: E402

from model_serving.api.app import create_app  # noqa: E402
from model_serving.config import ModelEntry, load_registry  # noqa: E402
from model_serving.engines.base import GenerationRequest  # noqa: E402

QWEN3_KWARGS = {"enable_thinking": False}
HI = [{"role": "user", "content": "hi"}]


@pytest.fixture(scope="module")
def engine(tiny_model_dir):
    from model_serving.engines.transformers_engine import TransformersEngine

    entry = ModelEntry(name="tiny", source=str(tiny_model_dir), chat_template_kwargs=QWEN3_KWARGS)
    return TransformersEngine.load(entry, tiny_model_dir, device="cpu")


def greedy(max_tokens=8, **kw):
    return GenerationRequest(messages=HI, max_tokens=max_tokens, temperature=0.0,
                             chat_template_kwargs=QWEN3_KWARGS, **kw)


# ------------------------------------------------------------ transformers engine

def test_generate_respects_max_tokens(engine):
    out = engine.generate(greedy(max_tokens=5))
    assert out.completion_tokens <= 5 and out.prompt_tokens > 0
    assert out.finish_reason in ("length", "stop")
    if out.completion_tokens == 5:
        assert out.finish_reason == "length"


def test_stream_matches_generate_when_greedy(engine):
    full = engine.generate(greedy())
    deltas = list(engine.stream(greedy()))
    assert "".join(d.text for d in deltas).strip() == full.text
    last = deltas[-1]
    assert (last.finish_reason, last.completion_tokens) == (full.finish_reason, full.completion_tokens)


class ScriptedModel:
    """Stands in for the network: emits fixed token ids, so stop handling is deterministic
    (the random tiny model only produces special tokens, which decode to "")."""

    def __init__(self, real, ids):
        self.device, self.ids = real.device, ids

    def generate(self, input_ids, streamer=None, **kwargs):
        import torch

        out = torch.cat([input_ids, torch.tensor([self.ids])], dim=1)
        if streamer is not None:
            streamer.put(input_ids[0])
            for token in self.ids:
                streamer.put(torch.tensor([token]))
            streamer.end()
        return out


@pytest.fixture
def scripted(engine):
    from model_serving.engines.transformers_engine import TransformersEngine

    ids = engine.tokenizer("the answer is 42 and more text", add_special_tokens=False)["input_ids"]
    return TransformersEngine(ScriptedModel(engine.model, ids), engine.tokenizer, model_name="tiny",
                              chat_template_kwargs=QWEN3_KWARGS)


def test_stop_string_cuts_the_output(scripted):
    out = scripted.generate(greedy(max_tokens=64, stop=["42"]))
    assert (out.text, out.finish_reason) == ("the answer is", "stop")
    deltas = list(scripted.stream(greedy(max_tokens=64, stop=["42"])))
    assert "".join(d.text for d in deltas).strip() == "the answer is"  # "4" was held back, never sent
    assert deltas[-1].finish_reason == "stop"
    assert scripted.generate(greedy(max_tokens=64)).text == "the answer is 42 and more text"


def test_prompt_uses_template_kwargs_and_tools(engine):
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]
    req = greedy(tools=tools)
    prompt = engine.tokenizer.decode(engine._inputs(req)["input_ids"][0])
    assert "<think>\n\n</think>" in prompt  # enable_thinking=False reached the template
    assert '"name": "lookup"' in prompt     # tools rendered into the system block


def test_template_kwargs_come_from_run_info_when_unset(tmp_path):
    from model_serving.engines.transformers_engine import template_kwargs_for

    (tmp_path / "run_info.json").write_text(json.dumps({"chat_template_kwargs": QWEN3_KWARGS}))
    assert template_kwargs_for(ModelEntry(name="m", source=str(tmp_path)), tmp_path) == QWEN3_KWARGS
    explicit = ModelEntry(name="m", source=str(tmp_path), chat_template_kwargs={})
    assert template_kwargs_for(explicit, tmp_path) == {}  # models.yaml wins
    assert template_kwargs_for(ModelEntry(name="m", source="x"), tmp_path / "missing") == {}


# --------------------------------------------------------------- OpenAI SDK client

@pytest.fixture(scope="module")
def client(engine):
    openai = pytest.importorskip("openai")
    http = TestClient(create_app(engine, max_tokens_limit=16))
    return openai.OpenAI(base_url="http://testserver/v1", api_key="unused", http_client=http)


def test_openai_sdk_chat_completion(client):
    resp = client.chat.completions.create(model="tiny", messages=HI, max_tokens=4, temperature=0)
    assert resp.choices[0].message.role == "assistant"
    assert resp.usage.completion_tokens <= 4
    assert [m.id for m in client.models.list().data] == ["tiny"]


def test_openai_sdk_streaming(client):
    chunks = list(client.chat.completions.create(model="tiny", messages=HI, max_tokens=4,
                                                 temperature=0, stream=True))
    assert chunks[-1].choices[0].finish_reason in ("length", "stop")
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    full = client.chat.completions.create(model="tiny", messages=HI, max_tokens=4, temperature=0)
    assert text.strip() == (full.choices[0].message.content or "")


def test_openai_sdk_rejects_unknown_model(client):
    import openai

    with pytest.raises(openai.NotFoundError):
        client.chat.completions.create(model="nope", messages=HI, max_tokens=4)


# ------------------------------------------------- consistency with finetune profiles

def test_qwen3_template_kwargs_match_the_training_profile():
    """Serving and training must render prompts identically (CLAUDE.md)."""
    from finetune.profiles import load_profiles, resolve_profile

    profiles = load_profiles()
    for entry in load_registry().values():
        if entry.chat_template_kwargs is None or entry.is_local:
            continue
        profile = resolve_profile(entry.source)
        if profile.name != "default":
            assert entry.chat_template_kwargs == profiles[profile.name].chat_template_kwargs, entry.name
