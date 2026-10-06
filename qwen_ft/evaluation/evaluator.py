"""Score models on held-out conversations.

The model loader is injected (any callable path -> object with `generate` and
`completion_loss`), so this module doesn't depend on torch or on how models are loaded.
"""
import gc
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

from qwen_ft.config import DEFAULT_BASE_MODEL
from qwen_ft.modeling.params import GenerationParams


@dataclass
class ExampleResult:
    loss: float   # mean per-token NLL of the reference answer (lower is better)
    answer: str   # what the model generated


@dataclass
class ModelEvaluation:
    model: str
    results: list[ExampleResult]

    @property
    def mean_loss(self):
        return statistics.mean(r.loss for r in self.results)


def base_model_of(model_path):
    """Read the base model id recorded by training, if any."""
    info = Path(model_path) / "run_info.json"
    if info.is_file():
        return json.loads(info.read_text())["base_model"]
    return DEFAULT_BASE_MODEL


def evaluate_model(load_model, model_path, conversations, params: GenerationParams) -> ModelEvaluation:
    chat = load_model(model_path)
    results = []
    for messages in conversations:
        prompt, reference = messages[:-1], messages[-1]["content"]
        results.append(ExampleResult(loss=chat.completion_loss(prompt, reference),
                                     answer=chat.generate(prompt, params)))
    del chat
    gc.collect()
    _free_gpu_cache()
    return ModelEvaluation(model=str(model_path), results=results)


def compare_models(load_model, base_path, tuned_path, conversations, params):
    """Models are loaded one at a time, so both don't have to fit in VRAM together."""
    return (evaluate_model(load_model, base_path, conversations, params),
            evaluate_model(load_model, tuned_path, conversations, params))


def _free_gpu_cache():
    torch = sys.modules.get("torch")  # only if the loader already imported it
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
