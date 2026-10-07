"""Contract check: does a server speak the OpenAI-compatible API the apps rely on?

Run it against this server, vLLM or Ollama before switching `*_BASE_URL`:

    python -m model_serving check --base-url http://localhost:8001/v1 --model qwen3-0.6b

Uses only the standard library (urllib). The HTTP function is injectable for tests.
"""
import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable

# (method, url, json body or None, headers) -> (status, response text)
Http = Callable[[str, str, dict | None, dict], tuple[int, str]]


def urllib_http(method: str, url: str, body: dict | None, headers: dict, timeout: float = 120) -> tuple[int, str]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


@dataclass
class CheckReport:
    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failed

    def format(self) -> str:
        lines = [f"PASS  {p}" for p in self.passed] + [f"FAIL  {f}" for f in self.failed]
        lines.append(f"\n{len(self.passed)} passed, {len(self.failed)} failed")
        return "\n".join(lines)


def run_checks(base_url: str, model: str, *, api_key: str | None = None, http: Http = urllib_http,
               chat_template_kwargs: dict | None = None) -> CheckReport:
    base = base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    report = CheckReport()

    def check(name, fn):
        try:
            fn()
            report.passed.append(name)
        except Exception as e:  # each check reports instead of aborting the run
            report.failed.append(f"{name}: {type(e).__name__}: {e}")

    def chat(extra):
        body = {"model": model, "messages": [{"role": "user", "content": "Say hello in one word."}],
                "max_tokens": 8, "temperature": 0, **extra}
        if chat_template_kwargs:
            body["chat_template_kwargs"] = chat_template_kwargs
        return http("POST", f"{base}/chat/completions", body, headers)

    def models_lists_model():
        status, text = http("GET", f"{base}/models", None, headers)
        assert status == 200, f"status {status}: {text[:200]}"
        ids = [m["id"] for m in json.loads(text)["data"]]
        assert model in ids, f"{model!r} not in {ids}"

    def completion_shape():
        status, text = chat({})
        assert status == 200, f"status {status}: {text[:200]}"
        data = json.loads(text)
        choice = data["choices"][0]
        assert choice["message"]["role"] == "assistant"
        assert isinstance(choice["message"]["content"], str)
        assert choice["finish_reason"] in ("stop", "length")
        assert data["usage"]["completion_tokens"] <= 8, "max_tokens was not respected"

    def no_thinking_block():
        status, text = chat({})
        content = json.loads(text)["choices"][0]["message"]["content"]
        assert "<think>" not in content, "model emitted a <think> block: set chat_template_kwargs"

    def streaming():
        status, text = chat({"stream": True})
        assert status == 200, f"status {status}"
        events = [line[6:] for line in text.splitlines() if line.startswith("data: ")]
        assert events and events[-1] == "[DONE]", "stream must end with data: [DONE]"
        chunks = [json.loads(e) for e in events[:-1]]
        assert all(c["object"] == "chat.completion.chunk" for c in chunks)
        assert any(c["choices"] and c["choices"][0].get("finish_reason") for c in chunks), "no finish_reason"

    def unknown_model_rejected():
        status, _ = http("POST", f"{base}/chat/completions",
                         {"model": "no-such-model", "messages": [{"role": "user", "content": "x"}],
                          "max_tokens": 1}, headers)
        assert status in (400, 404), f"expected 400/404, got {status}"

    check("GET /models lists the model", models_lists_model)
    check("chat completion has the OpenAI shape and respects max_tokens", completion_shape)
    check("no <think> block in the answer (chat_template_kwargs applied)", no_thinking_block)
    check("streaming uses SSE chunks and ends with [DONE]", streaming)
    check("unknown model name is rejected", unknown_model_rejected)
    return report
