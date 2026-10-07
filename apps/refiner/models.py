"""Model construction: OpenAI-compatible servers (vLLM, llama.cpp, LM Studio, OpenAI), Ollama,
or any local text-generation function (e.g. an in-process fine-tuned model).

Output modes (how a typed output is obtained):
    tool      the model calls an output tool (needs tool-calling support)
    native    the server enforces a JSON schema (response_format)
    prompted  the schema goes into the prompt and plain text is parsed: works with any model
"""
from dataclasses import dataclass
from typing import Callable, Literal

from pydantic_ai import NativeOutput, PromptedOutput, ToolOutput
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.settings import ModelSettings

OutputMode = Literal["tool", "native", "prompted"]

# Network-level failures that should move on to the next model in a fallback chain.
FALLBACK_ON = (ModelAPIError, ConnectionError, TimeoutError, OSError)


@dataclass(frozen=True)
class ModelSpec:
    provider: Literal["openai_compatible", "ollama"]
    model: str
    base_url: str | None = None
    api_key: str | None = None
    timeout: float = 60.0


def build_model(spec: ModelSpec) -> Model:
    from pydantic_ai.models.openai import OpenAIChatModel

    if spec.provider == "ollama":
        from pydantic_ai.providers.ollama import OllamaProvider

        provider = OllamaProvider(base_url=spec.base_url or "http://localhost:11434/v1")
    else:
        from pydantic_ai.providers.openai import OpenAIProvider

        # Local servers accept any key, but the OpenAI client refuses to start without one.
        provider = OpenAIProvider(base_url=spec.base_url, api_key=spec.api_key or "not-needed")
    return OpenAIChatModel(spec.model, provider=provider,
                           settings=ModelSettings(temperature=0.0, timeout=spec.timeout))


def with_fallbacks(models: list[Model]) -> Model:
    if not models:
        raise ValueError("at least one model is required")
    if len(models) == 1:
        return models[0]
    return FallbackModel(models[0], *models[1:], fallback_on=FALLBACK_ON)


def output_spec(output_type, mode: OutputMode):
    return {"tool": ToolOutput, "native": NativeOutput, "prompted": PromptedOutput}[mode](output_type)


def to_chat_messages(messages: list[ModelMessage], instructions: str | None) -> list[dict]:
    """Flatten PydanticAI messages into plain chat dicts for a text-only model."""
    chat: list[dict] = []
    if instructions:
        chat.append({"role": "system", "content": instructions})
    for message in messages:
        for part in message.parts:
            if isinstance(part, SystemPromptPart):
                chat.append({"role": "system", "content": part.content})
            elif isinstance(part, UserPromptPart):
                chat.append({"role": "user", "content": part.content if isinstance(part.content, str)
                             else str(part.content)})
            elif isinstance(part, RetryPromptPart):
                chat.append({"role": "user", "content": part.model_response()})
            elif isinstance(part, ToolReturnPart):
                chat.append({"role": "user", "content": part.model_response_str()})
            elif isinstance(part, TextPart):
                chat.append({"role": "assistant", "content": part.content})
            elif isinstance(part, ToolCallPart):
                chat.append({"role": "assistant", "content": part.args_as_json_str()})
    # Chat templates expect one leading system message: merge any extra ones into it.
    systems = [m["content"] for m in chat if m["role"] == "system"]
    rest = [m for m in chat if m["role"] != "system"]
    return ([{"role": "system", "content": "\n\n".join(systems)}] if systems else []) + rest


def text_function_model(generate: Callable[[list[dict]], str], name: str = "local-function") -> FunctionModel:
    """Wrap any `chat messages -> text` function (e.g. an in-process model) as a PydanticAI model.
    Use it with output mode "prompted": the function has no tool calling."""

    def call(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(generate(to_chat_messages(messages, info.instructions)))])

    return FunctionModel(call, model_name=name)

