"""Adapter: ops_crew's Refiner port implemented with the generic `refiner` package.

This is the only module that imports `refiner` (enforced by tests/test_architecture.py).
It supplies the ops-specific parts: prompts (with untrusted text delimited), the
correctable checks from domain/correction.py, and model construction from llms.yaml.
"""
import json
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ops_crew.crew.config import ConfigError, CrewConfig, LlmProfile
from ops_crew.domain.correction import correctable_problems
from ops_crew.domain.ports import RefinementError, ReflectionOutcome, RepairRequest
from ops_crew.schemas import ActionProposal, ResearchFindings
from refiner import (
    Critique,
    ModelSpec,
    RefinerError,
    build_model,
    make_critic,
    make_reviser,
    reflect,
    self_correct,
    text_function_model,
)
from refiner.models import OutputMode, with_fallbacks

ROLES = ("repair", "critic", "reviser")
_TAGS = ("user_request", "research", "previous_output", "answer", "critique", "example")
MAX_PREVIOUS_OUTPUT_CHARS = 4000


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReflectionConfig(_Strict):
    threshold: float = Field(ge=0, le=1)
    max_rounds: int = Field(ge=0, le=5)


class RefineInstructions(_Strict):
    repair: str
    critic_answer: str
    critic_example: str
    reviser: str


class RefineConfig(_Strict):
    max_retries: int = Field(ge=0, le=5)
    reflection: ReflectionConfig
    profiles: dict[str, str]
    instructions: RefineInstructions


def load_refine_config(config_dir: Path) -> RefineConfig:
    try:
        data = yaml.safe_load((Path(config_dir) / "refine.yaml").read_text(encoding="utf-8"))
        config = RefineConfig.model_validate(data)
    except (OSError, yaml.YAMLError, ValueError) as e:
        raise ConfigError(f"invalid refine config in {config_dir}: {e}") from e
    if set(config.profiles) != set(ROLES):
        raise ConfigError(f"refine.yaml profiles must define exactly {list(ROLES)}")
    return config


class RevisedAnswer(BaseModel):
    answer: str = Field(min_length=1, max_length=8000)


def _data(tag: str, text: str) -> str:
    """Delimit untrusted text and neutralise any of our tags inside it."""
    for t in _TAGS:
        text = text.replace(f"</{t}>", f"<\\/{t}>").replace(f"<{t}>", f"<\\{t}>")
    return f"<{tag}>\n{text}\n</{tag}>"


def _findings_json(findings: ResearchFindings) -> str:
    return findings.model_dump_json(indent=1)


# ------------------------------------------------------------------ models

def _api_key(profile: LlmProfile, env: dict) -> str | None:
    key = env.get(profile.api_key_env) if profile.api_key_env else None
    if key is None and profile.api_key_required:
        raise ConfigError(f"set {profile.api_key_env} to use model {profile.model}")
    return key


def default_output_mode(profile: LlmProfile) -> OutputMode:
    if profile.output_mode:
        return profile.output_mode
    return "prompted" if profile.provider == "finetuned" or profile.base_url else "tool"


def _local_generate(model_path: str, max_new_tokens: int):
    """Lazy in-process generation with this repo's fine-tuned model (prompted output only)."""
    state: dict[str, Any] = {}

    def generate(chat: list[dict]) -> str:
        from finetune.modeling.params import GenerationParams

        if "chat" not in state:
            from finetune.modeling.chat_model import ChatModel

            state["chat"] = ChatModel.load(model_path)
        return state["chat"].generate(chat, GenerationParams(temperature=0.0, max_new_tokens=max_new_tokens))

    return generate


def build_profile_model(profile: LlmProfile, env: dict):
    if profile.provider == "anthropic":
        raise ConfigError("anthropic profiles are not supported by the refiner (pydantic-ai's anthropic "
                          "extra conflicts with crewai's pin); use an OpenAI-compatible profile")
    if profile.provider == "finetuned":
        return text_function_model(_local_generate(profile.model, profile.max_tokens or 1024), name="finetuned")
    model_name = profile.model.split("/", 1)[1] if profile.model.startswith("openai/") else profile.model
    return build_model(ModelSpec(provider="openai_compatible", model=model_name, base_url=profile.base_url,
                                 api_key=_api_key(profile, env), timeout=profile.timeout))


def build_role_model(role: str, crew_config: CrewConfig, refine_config: RefineConfig, env: dict):
    """Model (with fallbacks) and output mode for one refiner role."""
    name = env.get(f"OPS_REFINE_PROFILE_{role.upper()}") or refine_config.profiles[role]
    if name not in crew_config.profiles:
        raise ConfigError(f"refiner role {role}: LLM profile {name!r} is not defined in llms.yaml")
    profile = crew_config.profiles[name]
    models = [build_profile_model(profile, env)]
    for fallback in profile.fallbacks:
        try:
            models.append(build_profile_model(crew_config.profiles[fallback], env))
        except ConfigError:
            continue  # not usable on this machine (no key) or not supported by the refiner
    return with_fallbacks(models), default_output_mode(profile)


# ----------------------------------------------------------------- refiner

class PydanticAIRefiner:
    """Implements ops_crew.domain.ports.Refiner."""

    def __init__(self, config: RefineConfig, models: dict[str, tuple[Any, OutputMode]]):
        self._config = config
        self._models = models

    @classmethod
    def from_config(cls, config_dir: Path, crew_config: CrewConfig, env: dict | None = None) -> "PydanticAIRefiner":
        env = dict(os.environ) if env is None else env
        refine_config = load_refine_config(config_dir)
        models = {role: build_role_model(role, crew_config, refine_config, env) for role in ROLES}
        return cls(refine_config, models)

    def _model(self, role):
        return self._models[role]

    # ---- self-correction
    def repair(self, req: RepairRequest) -> ActionProposal:
        model, mode = self._model("repair")
        prompt = "\n\n".join([
            "Problems to fix (from the validator):\n" + "\n".join(f"- {p}" for p in req.problems),
            _data("user_request", req.request),
            "Classification:\n" + req.classification.model_dump_json(),
            _data("research", _findings_json(req.findings)),
            _data("previous_output", req.previous_output[:MAX_PREVIOUS_OUTPUT_CHARS]),
            "Return the corrected ActionProposal.",
        ])

        def still_correctable(p: ActionProposal) -> list[str]:
            return correctable_problems(p, req.classification, req.findings, req.request)

        try:
            return self_correct(model, ActionProposal, prompt, instructions=self._config.instructions.repair,
                                checks=[still_correctable], max_retries=self._config.max_retries,
                                output_mode=mode).output
        except RefinerError as e:
            raise RefinementError(str(e)) from e

    # ---- reflection
    def review_answer(self, request: str, proposal: ActionProposal, findings: ResearchFindings) -> ReflectionOutcome:
        critic_model, critic_mode = self._model("critic")
        reviser_model, reviser_mode = self._model("reviser")
        context = _data("user_request", request) + "\n\n" + _data("research", _findings_json(findings))

        critic = make_critic(
            critic_model, instructions=self._config.instructions.critic_answer, output_mode=critic_mode,
            max_retries=self._config.max_retries,
            render=lambda d: f"{context}\n\n{_data('answer', d.answer)}",
        )
        reviser = make_reviser(
            reviser_model, RevisedAnswer, instructions=self._config.instructions.reviser, output_mode=reviser_mode,
            max_retries=self._config.max_retries,
            render=lambda d, v: f"{context}\n\n{_data('answer', d.answer)}\n\n{_data('critique', _critique_text(v))}",
        )
        result = reflect(RevisedAnswer(answer=proposal.answer or ""), critique=critic, revise=reviser,
                         threshold=self._config.reflection.threshold, max_rounds=self._config.reflection.max_rounds)
        best = proposal.model_copy(update={"answer": result.best.answer})
        return _outcome(best, result)

    def assess_example(self, request: str, proposal: ActionProposal, findings: ResearchFindings) -> ReflectionOutcome:
        critic_model, critic_mode = self._model("critic")
        example = json.dumps({"question": proposal.params.question, "answer": proposal.params.answer},
                             ensure_ascii=False, indent=1)
        critic = make_critic(critic_model, instructions=self._config.instructions.critic_example,
                             output_mode=critic_mode, max_retries=self._config.max_retries,
                             render=lambda _: _data("example", example))

        def never_rewrite(draft, verdict):  # the user's Q&A is data to judge, not to rewrite
            return draft

        result = reflect(proposal, critique=critic, revise=never_rewrite,
                         threshold=self._config.reflection.threshold, max_rounds=0)
        return _outcome(proposal, result)


def _critique_text(c: Critique) -> str:
    return json.dumps({"score": c.score, "grounded": c.grounded, "issues": c.issues}, ensure_ascii=False)


def _outcome(proposal: ActionProposal, result) -> ReflectionOutcome:
    issues = next((r.issues for r in reversed(result.rounds) if r.output == result.best), [])
    return ReflectionOutcome(proposal=proposal, score=result.best_score, accepted=result.accepted,
                             stop_reason=result.stop_reason, rounds=max(len(result.rounds) - 1, 0), issues=issues)
