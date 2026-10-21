"""Multi-LLM support: every agent gets an LLM built from a named profile.

Selection order per agent: OPS_LLM_PROFILE_<AGENT> > OPS_LLM_PROFILE > agents.yaml `llm`.
Each profile may list fallbacks, tried in order if the provider call raises.
Temperature is always 0.
"""
from typing import Any, Callable, Protocol

from crewai import LLM
from crewai.llms.base_llm import BaseLLM
from pydantic import Field, PrivateAttr

from ops_crew.crew.config import ConfigError, CrewConfig, LlmProfile


class LLMFactory(Protocol):
    def for_agent(self, agent_name: str) -> BaseLLM: ...


def profile_name_for(agent_name: str, config: CrewConfig, env: dict) -> str:
    name = (env.get(f"OPS_LLM_PROFILE_{agent_name.upper()}") or env.get("OPS_LLM_PROFILE")
            or config.agents[agent_name].llm)
    if name not in config.profiles:
        raise ConfigError(f"agent {agent_name}: LLM profile {name!r} is not defined in llms.yaml")
    return name


class FallbackLLM(BaseLLM):
    """Calls the primary LLM; on any exception tries each fallback in order."""
    chain: list[Any] = Field(default_factory=list)
    _on_fallback: Callable[[str, Exception], None] | None = PrivateAttr(default=None)

    def call(self, messages, tools=None, callbacks=None, available_functions=None, from_task=None,
             from_agent=None, response_model=None):
        errors = []
        for llm in self.chain:
            try:
                return llm.call(messages, tools=tools, callbacks=callbacks, available_functions=available_functions,
                                from_task=from_task, from_agent=from_agent, response_model=response_model)
            except Exception as e:
                errors.append(f"{llm.model}: {type(e).__name__}: {e}")
                if self._on_fallback:
                    self._on_fallback(llm.model, e)
        raise RuntimeError("all LLM providers failed: " + " | ".join(errors))

    def supports_function_calling(self) -> bool:
        return all(llm.supports_function_calling() for llm in self.chain)

    def supports_stop_words(self) -> bool:
        return all(llm.supports_stop_words() for llm in self.chain)

    def get_context_window_size(self) -> int:
        return min(llm.get_context_window_size() for llm in self.chain)


class FinetunedLLM(BaseLLM):
    """In-process adapter for this repo's fine-tuned Qwen (finetune.modeling.ChatModel). Loaded lazily."""
    max_new_tokens: int = 1024
    _chat: Any = PrivateAttr(default=None)

    def call(self, messages, tools=None, callbacks=None, available_functions=None, from_task=None,
             from_agent=None, response_model=None):
        from finetune.modeling.params import GenerationParams

        if self._chat is None:
            from finetune.modeling.chat_model import ChatModel

            self._chat = ChatModel.load(self.model)
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        chat_messages = [{"role": m["role"], "content": m["content"] or ""} for m in messages]
        return self._chat.generate(chat_messages, GenerationParams(temperature=0.0, max_new_tokens=self.max_new_tokens))

    def supports_function_calling(self) -> bool:
        return False


def _api_key(profile: LlmProfile, env: dict) -> str | None:
    key = env.get(profile.api_key_env) if profile.api_key_env else None
    if key is None and profile.api_key_required:
        raise ConfigError(f"set {profile.api_key_env} to use model {profile.model}")
    return key


def build_llm(profile: LlmProfile, env: dict) -> BaseLLM:
    if profile.provider == "finetuned":
        return FinetunedLLM(model=profile.model, temperature=0.0, max_new_tokens=profile.max_tokens or 1024)
    kwargs = {"model": profile.model, "temperature": 0.0, "timeout": profile.timeout}
    key = _api_key(profile, env)
    # Local servers accept any key, but the OpenAI client refuses to start without one.
    kwargs["api_key"] = key or ("not-needed" if profile.provider == "openai_compatible" else None)
    if profile.base_url:
        kwargs["base_url"] = profile.base_url
    if profile.max_tokens:
        kwargs["max_tokens"] = profile.max_tokens
    return LLM(**{k: v for k, v in kwargs.items() if v is not None})


class ProfileLLMFactory:
    def check(self) -> None:
        """Build every agent's LLM once so a missing key or bad profile fails at startup,
        not as a retried crew failure on the first request."""
        for agent_name in self._config.agents:
            self.for_agent(agent_name)

    def __init__(self, config: CrewConfig, env: dict,
                 on_fallback: Callable[[str, Exception], None] | None = None):
        self._config = config
        self._env = env
        self._on_fallback = on_fallback

    def for_agent(self, agent_name: str) -> BaseLLM:
        name = profile_name_for(agent_name, self._config, self._env)
        profile = self._config.profiles[name]
        primary = build_llm(profile, self._env)
        chain = [primary]
        for fallback_name in profile.fallbacks:
            try:
                chain.append(build_llm(self._config.profiles[fallback_name], self._env))
            except ConfigError:
                continue  # fallback not configured on this machine (e.g. no API key): skip it
        if len(chain) == 1:
            return primary
        llm = FallbackLLM(model=" -> ".join(p.model for p in chain), temperature=0.0, chain=chain)
        llm._on_fallback = self._on_fallback
        return llm
