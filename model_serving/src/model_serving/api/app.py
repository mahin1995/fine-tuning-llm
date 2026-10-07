"""FastAPI app implementing the OpenAI-compatible contract (ARCHITECTURE.md 3.3).

    GET  /health
    GET  /v1/models
    POST /v1/chat/completions   (stream and non-stream, tools via the chat template)

The engine is injected, so tests use FakeEngine and the CLI wires TransformersEngine.
Generation policy (max_tokens, temperature) comes from each request; the server only caps
max_tokens at the model's max_tokens_limit.
"""
import hmac
import json
import time
import uuid

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from model_serving.api.schemas import AssistantMessage, ChatCompletion, ChatCompletionRequest, Choice, Usage
from model_serving.engines.base import GenerationRequest, InferenceEngine
from model_serving.engines.toolcalls import parse_tool_calls, to_template_messages


class ApiError(Exception):
    def __init__(self, status: int, message: str, type_: str = "invalid_request_error", code: str | None = None):
        self.status, self.message, self.type, self.code = status, message, type_, code


def _error(status, message, type_="invalid_request_error", code=None):
    return JSONResponse({"error": {"message": message, "type": type_, "code": code}}, status_code=status)


def _flatten_content(content) -> str:
    if content is None or isinstance(content, str):
        return content or ""
    texts = []
    for part in content:
        if part.get("type") != "text":
            raise ApiError(400, f"content part type {part.get('type')!r} is not supported (text only)")
        texts.append(part.get("text", ""))
    return "".join(texts)


def to_engine_messages(body: ChatCompletionRequest) -> list[dict]:
    messages = []
    for m in body.messages:
        message = {"role": "system" if m.role == "developer" else m.role, "content": _flatten_content(m.content)}
        if m.tool_calls:
            message["tool_calls"] = [c.model_dump() for c in m.tool_calls]
        if m.tool_call_id:
            message["tool_call_id"] = m.tool_call_id
        messages.append(message)
    return to_template_messages(messages)


def create_app(engine: InferenceEngine, *, max_tokens_limit: int, api_key: str | None = None) -> FastAPI:
    app = FastAPI(title="model_serving", docs_url=None, redoc_url=None)

    @app.exception_handler(ApiError)
    async def api_error(request, exc: ApiError):
        return _error(exc.status, exc.message, exc.type, exc.code)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc: RequestValidationError):
        problems = "; ".join(f"{'.'.join(map(str, e['loc'][1:])) or 'body'}: {e['msg']}" for e in exc.errors())
        return _error(400, problems)

    def authorize(request: Request):
        if api_key is None:
            return
        given = request.headers.get("authorization", "")
        if not hmac.compare_digest(given.encode(), f"Bearer {api_key}".encode()):
            raise ApiError(401, "invalid or missing API key", "authentication_error", "invalid_api_key")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": engine.model_name}

    @app.get("/v1/models", dependencies=[Depends(authorize)])
    def models():
        return {"object": "list", "data": [{"id": engine.model_name, "object": "model", "created": 0,
                                            "owned_by": "model_serving", "max_tokens_limit": max_tokens_limit}]}

    def build_request(body: ChatCompletionRequest) -> tuple[GenerationRequest, bool]:
        if body.model != engine.model_name:
            raise ApiError(404, f"model {body.model!r} is not served here (serving {engine.model_name!r})",
                           code="model_not_found")
        if body.n != 1:
            raise ApiError(400, "only n=1 is supported")
        if isinstance(body.tool_choice, dict) or body.tool_choice == "required":
            raise ApiError(400, "tool_choice must be 'auto' or 'none' (forced tool calls are not supported)")
        requested = body.max_tokens or body.max_completion_tokens or max_tokens_limit
        tools = body.tools if body.tools and body.tool_choice != "none" else None
        return GenerationRequest(
            messages=to_engine_messages(body),
            max_tokens=min(requested, max_tokens_limit),
            temperature=body.temperature,
            top_p=body.top_p,
            stop=body.stop_list(),
            tools=tools,
            chat_template_kwargs={**engine.default_chat_template_kwargs, **(body.chat_template_kwargs or {})},
        ), requested > max_tokens_limit

    def headers(limited: bool):
        return {"X-Max-Tokens-Limited": "true"} if limited else {}

    @app.post("/v1/chat/completions", dependencies=[Depends(authorize)])
    def chat_completions(body: ChatCompletionRequest):
        req, limited = build_request(body)
        completion_id, created = f"chatcmpl-{uuid.uuid4().hex[:24]}", int(time.time())
        if body.stream:
            include_usage = bool((body.stream_options or {}).get("include_usage"))
            return StreamingResponse(_sse(engine, req, completion_id, created, include_usage),
                                     media_type="text/event-stream", headers=headers(limited))
        gen = engine.generate(req)
        content, calls = parse_tool_calls(gen.text) if req.tools else (gen.text, [])
        completion = ChatCompletion(
            id=completion_id, created=created, model=engine.model_name,
            choices=[Choice(message=AssistantMessage(content=(content or None) if calls else content,
                                                     tool_calls=calls or None),
                            finish_reason="tool_calls" if calls else gen.finish_reason)],
            usage=Usage(prompt_tokens=gen.prompt_tokens, completion_tokens=gen.completion_tokens,
                        total_tokens=gen.prompt_tokens + gen.completion_tokens),
        )
        return JSONResponse(completion.model_dump(exclude_none=True), headers=headers(limited))

    return app


def _chunk(completion_id, created, model, delta, finish_reason=None):
    return {"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}


def _event(payload) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _sse(engine, req: GenerationRequest, completion_id, created, include_usage):
    model = engine.model_name
    yield _event(_chunk(completion_id, created, model, {"role": "assistant", "content": ""}))
    if req.tools:
        # Tool calls can only be recognised in the complete output: generate, then send it at once.
        gen = engine.generate(req)
        content, calls = parse_tool_calls(gen.text)
        if calls:
            delta = {"tool_calls": [{"index": i, **c} for i, c in enumerate(calls)]}
            if content:
                delta["content"] = content
            yield _event(_chunk(completion_id, created, model, delta))
            finish = "tool_calls"
        else:
            yield _event(_chunk(completion_id, created, model, {"content": content}))
            finish = gen.finish_reason
        prompt_tokens, completion_tokens = gen.prompt_tokens, gen.completion_tokens
    else:
        finish, prompt_tokens, completion_tokens = "stop", 0, 0
        for delta in engine.stream(req):
            if delta.text:
                yield _event(_chunk(completion_id, created, model, {"content": delta.text}))
            if delta.finish_reason:
                finish, prompt_tokens, completion_tokens = (delta.finish_reason, delta.prompt_tokens,
                                                            delta.completion_tokens)
    yield _event(_chunk(completion_id, created, model, {}, finish))
    if include_usage:
        yield _event({"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": model,
                      "choices": [], "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                                               "total_tokens": prompt_tokens + completion_tokens}})
    yield "data: [DONE]\n\n"
