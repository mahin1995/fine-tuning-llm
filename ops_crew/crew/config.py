"""Load and validate the YAML config. Typos fail at startup, not halfway through a request."""
import os
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ops_crew import schemas
from ops_crew.domain.validation import TASK_SCHEMAS

_ENV_REF = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")


class ConfigError(ValueError):
    pass


def expand_env(value: str, env: dict) -> str:
    def replace(match):
        name, default = match.group(1), match.group(2)
        if name in env:
            return env[name]
        if default is None:
            raise ConfigError(f"environment variable {name} is not set")
        return default

    return _ENV_REF.sub(replace, value)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LlmProfile(_Strict):
    provider: Literal["openai_compatible", "anthropic", "qwen_ft"]
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    api_key_required: bool = True
    timeout: float = Field(default=60, gt=0)
    max_tokens: int | None = Field(default=None, gt=0)
    fallbacks: list[str] = Field(default_factory=list)


class AgentSpec(_Strict):
    role: str
    goal: str
    backstory: str
    llm: str
    tools: list[str] = Field(default_factory=list)
    max_iter: int = Field(ge=1, le=10)
    max_execution_time: int = Field(ge=1, le=600)


class TaskSpec(_Strict):
    agent: str
    output: str
    description: str
    expected_output: str
    context: list[str] = Field(default_factory=list)


class CrewConfig(_Strict):
    agents: dict[str, AgentSpec]
    tasks: dict[str, TaskSpec]
    profiles: dict[str, LlmProfile]

    @model_validator(mode="after")
    def cross_check(self):
        from ops_crew.crew.tools import TOOL_NAMES

        if list(self.tasks) != list(TASK_SCHEMAS):
            raise ValueError(f"tasks must be exactly {list(TASK_SCHEMAS)} in order, got {list(self.tasks)}")
        for name, task in self.tasks.items():
            if task.agent not in self.agents:
                raise ValueError(f"task {name}: unknown agent {task.agent!r}")
            if getattr(schemas, task.output, None) is not TASK_SCHEMAS[name]:
                raise ValueError(f"task {name}: output must be {TASK_SCHEMAS[name].__name__}")
            earlier = list(self.tasks)[: list(self.tasks).index(name)]
            for ctx in task.context:
                if ctx not in earlier:
                    raise ValueError(f"task {name}: context {ctx!r} must be an earlier task")
        for name, agent in self.agents.items():
            if agent.llm not in self.profiles:
                raise ValueError(f"agent {name}: unknown llm profile {agent.llm!r}")
            unknown = set(agent.tools) - TOOL_NAMES
            if unknown:
                raise ValueError(f"agent {name}: unknown tool(s) {sorted(unknown)} (hallucinated in config?)")
        for name, profile in self.profiles.items():
            missing = [f for f in profile.fallbacks if f not in self.profiles or f == name]
            if missing:
                raise ValueError(f"llm profile {name}: invalid fallback(s) {missing}")
        return self


def _read_yaml(path: Path) -> dict:
    with Path(path).open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping")
    return data


def _expand_profiles(profiles: dict, env: dict) -> dict:
    return {name: {k: expand_env(v, env) if isinstance(v, str) else v for k, v in p.items()}
            for name, p in profiles.items()}


def load_config(config_dir: Path, env: dict | None = None) -> CrewConfig:
    env = dict(os.environ) if env is None else env
    config_dir = Path(config_dir)
    try:
        return CrewConfig(
            agents=_read_yaml(config_dir / "agents.yaml"),
            tasks=_read_yaml(config_dir / "tasks.yaml"),
            profiles=_expand_profiles(_read_yaml(config_dir / "llms.yaml")["profiles"], env),
        )
    except (OSError, KeyError, yaml.YAMLError, ValueError) as e:
        raise ConfigError(f"invalid crew config in {config_dir}: {e}") from e
