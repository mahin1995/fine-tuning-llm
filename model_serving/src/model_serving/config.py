"""Model registry (models.yaml). Light module: no torch / transformers."""
import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

COMPONENT_DIR = Path(__file__).resolve().parents[2]          # model_serving/
DEFAULT_CONFIG = Path(os.environ.get("MODEL_SERVING_CONFIG", COMPONENT_DIR / "models.yaml"))
DEFAULT_MAX_TOKENS_LIMIT = 2048


class ConfigError(ValueError):
    pass


class ModelEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    source: str
    chat_template_kwargs: dict | None = None  # None: read from the model's run_info.json, else {}
    max_tokens_limit: int = Field(default=DEFAULT_MAX_TOKENS_LIMIT, ge=1, le=131072)
    dtype: Literal["auto", "bfloat16", "float16", "float32"] = "auto"
    revision: str | None = None

    @property
    def is_local(self) -> bool:
        return self.source.startswith(("/", "./", "../")) or Path(self.source).is_dir()


def load_registry(path: Path = DEFAULT_CONFIG) -> dict[str, ModelEntry]:
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        models = data["models"]
        return {name: ModelEntry(name=name, **(spec or {})) for name, spec in models.items()}
    except (OSError, KeyError, TypeError, yaml.YAMLError, ValueError) as e:
        raise ConfigError(f"invalid model registry {path}: {e}") from e


def resolve_entry(model: str, registry: dict[str, ModelEntry] | None = None) -> ModelEntry:
    """A registry name, or ad hoc: a Hugging Face id / directory with default settings."""
    registry = load_registry() if registry is None else registry
    if model in registry:
        return registry[model]
    name = Path(model).name if Path(model).is_dir() else model
    return ModelEntry(name=name, source=model)
