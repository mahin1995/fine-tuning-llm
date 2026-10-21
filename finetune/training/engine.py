"""Training engine port: one interface, swappable implementations (TRL today, Unsloth later).

Light module (no torch): the CLI and tests import it without loading a training stack.
Everything every engine must agree on lives here: profile resolution, the result type and
the run_info.json contract that evaluation and serving read.
"""
import importlib
import json
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Protocol

from finetune.profiles import ModelProfile, get_profile, resolve_profile
from finetune.training.options import TrainOptions

# name -> "module:Class". Add an engine here (e.g. "unsloth": "finetune.training.unsloth_engine:UnslothEngine").
ENGINES = {
    "trl": "finetune.training.trl_engine:TrlEngine",
}
DEFAULT_ENGINE = "trl"


class TrainingOutOfMemory(RuntimeError):
    """The GPU ran out of memory; the message says which options to change."""


@dataclass
class TrainResult:
    output_dir: str
    optimizer_steps: int
    metrics: dict
    engine: str
    profile: str
    merged_dir: str | None = None


class TrainingEngine(Protocol):
    name: str

    def train(self, opts: TrainOptions, profile: ModelProfile) -> TrainResult:
        """Train and save a model artifact (weights/adapter, tokenizer, run_info.json) to opts.output."""
        ...


def engine_names() -> list[str]:
    return list(ENGINES)


def get_engine(name: str) -> TrainingEngine:
    if name not in ENGINES:
        raise ValueError(f"unknown training engine {name!r}; available: {', '.join(ENGINES)}")
    module_name, class_name = ENGINES[name].split(":")
    return getattr(importlib.import_module(module_name), class_name)()


def profile_for(opts: TrainOptions) -> ModelProfile:
    """An explicit `opts.profile` wins; otherwise the profile is resolved from the base model."""
    return get_profile(opts.profile) if opts.profile else resolve_profile(opts.model)


def run_training(opts: TrainOptions, engine: str = DEFAULT_ENGINE) -> TrainResult:
    return get_engine(engine).train(opts, profile_for(opts))


def _versions(packages) -> dict:
    found = {}
    for pkg in packages:
        try:
            found[pkg] = version(pkg)
        except PackageNotFoundError:
            continue
    return found


def write_run_info(opts: TrainOptions, *, engine: str, profile: ModelProfile, optimizer_steps: int,
                   metrics: dict, packages=("torch", "transformers", "trl", "peft", "datasets")) -> Path:
    """The model-artifact contract (ARCHITECTURE.md 3.2). `profile` lets evaluation and serving
    render prompts exactly as training did, even for a local output directory."""
    info = {
        "base_model": opts.model,
        "method": opts.method,
        "engine": engine,
        "profile": profile.name,
        "chat_template_kwargs": profile.chat_template_kwargs,
        "options": asdict(opts),
        "optimizer_steps": optimizer_steps,
        "metrics": metrics,
        "versions": _versions(packages),
    }
    path = Path(opts.output) / "run_info.json"
    path.write_text(json.dumps(info, indent=2, default=str))
    return path
