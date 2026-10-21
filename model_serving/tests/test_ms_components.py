"""Tool-call parsing, registry, download, vLLM command and the contract check."""
import json

import pytest
from fastapi.testclient import TestClient

from model_serving.api.app import create_app
from model_serving.backends import vllm_command
from model_serving.check import run_checks
from model_serving.config import ConfigError, ModelEntry, load_registry, resolve_entry
from model_serving.download import DownloadError, fetch
from model_serving.engines.base import cut_at_stop
from model_serving.engines.fake import FakeEngine
from model_serving.engines.toolcalls import parse_tool_calls, to_template_messages


def test_parse_tool_calls():
    content, calls = parse_tool_calls('Let me look.\n<tool_call>\n{"name": "a", "arguments": {"x": 1}}\n</tool_call>')
    assert content == "Let me look."
    assert [c["function"] for c in calls] == [{"name": "a", "arguments": '{"x": 1}'}]
    two = parse_tool_calls('<tool_call>{"name": "a"}</tool_call><tool_call>{"name": "b", "arguments": "{}"}')
    assert [c["function"]["name"] for c in two[1]] == ["a", "b"]  # unclosed last tag tolerated


@pytest.mark.parametrize("text", [
    "plain answer with {\"name\": \"a\"} json",            # no tags: never a call
    '<tool_call>{broken</tool_call>',                       # malformed
    '<tool_call>{"name": "a"}</tool_call><tool_call>{"x": 1}</tool_call>',  # one bad block -> no calls
])
def test_no_guessed_tool_calls(text):
    assert parse_tool_calls(text) == (text, [])


def test_to_template_messages():
    out = to_template_messages([{"role": "assistant", "content": None, "tool_calls": [
        {"id": "1", "type": "function", "function": {"name": "a", "arguments": "not json"}}]}])
    assert out[0]["content"] == "" and out[0]["tool_calls"][0]["function"]["arguments"] == "not json"


def test_cut_at_stop():
    assert cut_at_stop("abc END def", ["END", "de"]) == ("abc ", True)
    assert cut_at_stop("abc", []) == ("abc", False)


def test_repo_registry():
    registry = load_registry()
    assert registry["qwen3-0.6b"].chat_template_kwargs == {"enable_thinking": False}
    assert registry["qwen3-ft"].chat_template_kwargs is None  # read from run_info.json
    assert all(e.max_tokens_limit >= 1 for e in registry.values())


def test_invalid_registry(tmp_path):
    bad = tmp_path / "m.yaml"
    bad.write_text("models: {x: {source: a, temperature: 1}}")
    with pytest.raises(ConfigError, match="temperature"):
        load_registry(bad)


def test_resolve_entry_ad_hoc(tmp_path):
    assert resolve_entry("Qwen/Qwen3-1.7B", {}).name == "Qwen/Qwen3-1.7B"
    entry = resolve_entry(str(tmp_path), {})
    assert entry.name == tmp_path.name and entry.is_local


def test_fetch_local_dir_is_checked_not_copied(tmp_path):
    with pytest.raises(DownloadError, match="not a model directory"):
        fetch(ModelEntry(name="x", source=str(tmp_path)))
    (tmp_path / "config.json").write_text("{}")
    assert fetch(ModelEntry(name="x", source=str(tmp_path))) == tmp_path


def test_vllm_command():
    cmd = vllm_command(ModelEntry(name="qwen3-ft", source="/models/outputs/qwen3-ft"), port=9000)
    assert cmd[cmd.index("-p") + 1] == "127.0.0.1:9000:8000"
    assert cmd[cmd.index("--served-model-name") + 1] == "qwen3-ft"
    assert all(not part.startswith("~") for part in cmd)
    assert any(part.endswith(":/models/outputs:ro") and part.startswith("/") for part in cmd)
    hub = vllm_command(ModelEntry(name="q", source="Qwen/Qwen3-0.6B"))
    assert not any(":/models/outputs" in part for part in hub)


def http_via(client):
    def http(method, url, body, headers):
        r = client.request(method, url, json=body, headers=headers)
        return r.status_code, r.text
    return http


def test_contract_check_passes_against_this_server():
    app = create_app(FakeEngine(model_name="m"), max_tokens_limit=8)
    report = run_checks("http://testserver/v1", "m", http=http_via(TestClient(app)))
    assert report.ok, report.format()
    assert len(report.passed) == 5


def test_contract_check_reports_a_broken_server():
    """A server that ignores max_tokens and thinks out loud fails exactly those checks."""
    thinker = FakeEngine(model_name="m", replies=["<think> hmm </think> " + "word " * 50])
    thinker.generate.__func__  # noqa: B018 (plain FakeEngine, limit below is what breaks it)
    app = create_app(thinker, max_tokens_limit=100)
    report = run_checks("http://testserver/v1", "m", http=http_via(TestClient(app)))
    assert not report.ok
    failed = " | ".join(report.failed)
    assert "<think> block" in failed
    assert len(report.failed) == 1  # FakeEngine still respects the requested max_tokens


def test_check_report_format():
    report = run_checks("http://x/v1", "m", http=lambda *a: (500, json.dumps({"error": {"message": "boom"}})))
    assert not report.ok and "0 passed, 5 failed" in report.format()
