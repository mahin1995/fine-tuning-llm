"""Deterministic engine for tests and for exercising clients without a GPU.

Replies are scripted (in order, last one repeats) or echo the last user message. One "token"
is one whitespace-separated word, so max_tokens and usage are easy to assert.
"""
from model_serving.engines.base import Delta, Generation, GenerationRequest, cut_at_stop


class FakeEngine:
    def __init__(self, model_name="fake-model", replies=None, default_chat_template_kwargs=None):
        self.model_name = model_name
        self.default_chat_template_kwargs = dict(default_chat_template_kwargs or {})
        self.replies = list(replies or [])
        self.requests: list[GenerationRequest] = []

    def _reply(self, req):
        if self.replies:
            return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        last_user = next((m["content"] for m in reversed(req.messages) if m["role"] == "user"), "")
        return f"echo: {last_user}"

    def generate(self, req: GenerationRequest) -> Generation:
        self.requests.append(req)
        words = self._reply(req).split(" ")
        finish = "length" if len(words) > req.max_tokens else "stop"
        text, stopped = cut_at_stop(" ".join(words[: req.max_tokens]), req.stop)
        prompt_tokens = sum(len(str(m.get("content", "")).split()) for m in req.messages)
        return Generation(text, prompt_tokens, len(text.split()) if text else 0, "stop" if stopped else finish)

    def stream(self, req: GenerationRequest):
        result = self.generate(req)
        words = result.text.split(" ") if result.text else []
        for i, word in enumerate(words):
            yield Delta(word if i == 0 else " " + word)
        yield Delta("", result.finish_reason, result.prompt_tokens, result.completion_tokens)
