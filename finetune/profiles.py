"""Model profiles: per-model-family settings that training and inference must share.

The chat template variables (e.g. Qwen3's enable_thinking=False) and the LoRA target layers
depend on the model family, not on our code. They live in model_profiles.yaml and are
resolved from the model id or directory, so another model family needs a YAML entry, not a
code change. Light module: no torch / transformers imports.
"""
import fnmatch
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

PROFILES_PATH = Path(__file__).parent / "model_profiles.yaml"
DEFAULT_PROFILE = "default"
ALL_LINEAR = "all-linear"


class ProfileError(ValueError):
    pass


@dataclass(frozen=True)
class ModelProfile:
    name: str
    chat_template_kwargs: dict = field(default_factory=dict)
    lora_target_modules: tuple[str, ...] | str = ALL_LINEAR
    model_types: tuple[str, ...] = ()
    hub_ids: tuple[str, ...] = ()

    def lora_targets(self):
        """In the form PEFT expects: a list of layer names, or the string "all-linear"."""
        return self.lora_target_modules if isinstance(self.lora_target_modules, str) \
            else list(self.lora_target_modules)


def _parse(name: str, raw: dict) -> ModelProfile:
    allowed = {"model_types", "hub_ids", "chat_template_kwargs", "lora_target_modules"}
    unknown = set(raw) - allowed
    if unknown:
        raise ProfileError(f"profile {name}: unknown key(s) {sorted(unknown)}")
    targets = raw.get("lora_target_modules", ALL_LINEAR)
    if isinstance(targets, str):
        if targets != ALL_LINEAR:
            raise ProfileError(f"profile {name}: lora_target_modules must be a list or '{ALL_LINEAR}'")
    elif not (isinstance(targets, list) and targets and all(isinstance(t, str) for t in targets)):
        raise ProfileError(f"profile {name}: lora_target_modules must be a non-empty list of names")
    kwargs = raw.get("chat_template_kwargs") or {}
    if not isinstance(kwargs, dict):
        raise ProfileError(f"profile {name}: chat_template_kwargs must be a mapping")
    return ModelProfile(
        name=name,
        chat_template_kwargs=dict(kwargs),
        lora_target_modules=targets if isinstance(targets, str) else tuple(targets),
        model_types=tuple(raw.get("model_types") or ()),
        hub_ids=tuple(raw.get("hub_ids") or ()),
    )


@lru_cache(maxsize=8)
def load_profiles(path: Path = PROFILES_PATH) -> dict[str, ModelProfile]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    raw_profiles = data.get("profiles")
    if not isinstance(raw_profiles, dict) or DEFAULT_PROFILE not in raw_profiles:
        raise ProfileError(f"{path}: needs a `profiles` mapping with a '{DEFAULT_PROFILE}' entry")
    return {name: _parse(name, raw or {}) for name, raw in raw_profiles.items()}


def get_profile(name: str, profiles: dict[str, ModelProfile] | None = None) -> ModelProfile:
    profiles = profiles or load_profiles()
    if name not in profiles:
        raise ProfileError(f"unknown model profile {name!r}; known: {', '.join(profiles)}")
    return profiles[name]


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def resolve_profile(model: str, profiles: dict[str, ModelProfile] | None = None) -> ModelProfile:
    """Profile for a Hub id, a model directory, a fine-tuned output or a LoRA adapter."""
    profiles = profiles or load_profiles()
    path = Path(model)
    if path.is_dir():
        recorded = _read_json(path / "run_info.json").get("profile")
        if recorded:
            return get_profile(recorded, profiles)
        base = _read_json(path / "adapter_config.json").get("base_model_name_or_path")
        if base and base != model:
            return resolve_profile(base, profiles)
        model_type = _read_json(path / "config.json").get("model_type")
        for profile in profiles.values():
            if model_type and model_type in profile.model_types:
                return profile
    for profile in profiles.values():
        if any(fnmatch.fnmatch(model.lower(), pattern.lower()) for pattern in profile.hub_ids):
            return profile
    return profiles[DEFAULT_PROFILE]
