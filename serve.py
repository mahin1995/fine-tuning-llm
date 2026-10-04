"""FastAPI chat server with a minimal browser UI.

    PORT=8000 ./run.sh python serve.py --model outputs/qwen3-ft
    # then open http://localhost:8000

Endpoints:
    GET  /         chat page (static/index.html)
    GET  /health   {"status": "ok", "model": ...}
    POST /chat     {"messages": [{"role": "user", "content": "..."}], "max_new_tokens": 512, "temperature": 0.7}
                   -> {"reply": "..."}
"""
import argparse
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from data_utils import DataError, validate_messages
from inference import ChatModel, GenerationParams

STATIC_DIR = Path(__file__).parent / "static"
MAX_MESSAGES = 40
MAX_CONTENT_CHARS = 8000


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=MAX_CONTENT_CHARS)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=MAX_MESSAGES)
    max_new_tokens: int = Field(default=512, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)

    @field_validator("messages")
    @classmethod
    def must_end_with_user(cls, messages):
        if messages[-1].role != "user":
            raise ValueError("last message must be from the user")
        return messages


class ChatResponse(BaseModel):
    reply: str


def create_app(model_loader):
    """`model_loader` is a zero-arg callable returning a ChatModel (injected so tests can fake it)."""

    @asynccontextmanager
    async def lifespan(app):
        app.state.chat = model_loader()
        yield

    app = FastAPI(title="Qwen fine-tuned chat", lifespan=lifespan)

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    def health(request: Request):
        return {"status": "ok", "model": request.app.state.chat.name}

    # Plain `def`: FastAPI runs it in a worker thread, so blocking generation doesn't stall the event loop.
    @app.post("/chat", response_model=ChatResponse)
    def chat(body: ChatRequest, request: Request):
        messages = [m.model_dump() for m in body.messages]
        try:
            # Same rules as the training data, except the conversation ends with a user turn.
            validate_messages(messages + [{"role": "assistant", "content": "-"}], "request")
        except DataError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        params = GenerationParams(max_new_tokens=body.max_new_tokens, temperature=body.temperature)
        return ChatResponse(reply=request.app.state.chat.generate(messages, params))

    return app


def main(argv=None):
    import uvicorn

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="outputs/qwen3-ft")
    # 0.0.0.0 is needed inside Docker; run.sh only publishes the port on the host's 127.0.0.1.
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)

    app = create_app(lambda: ChatModel.load(args.model))
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
