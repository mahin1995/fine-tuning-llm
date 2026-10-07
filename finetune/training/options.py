"""Training configuration, independent of how it was collected (CLI, tests, notebooks)."""
from dataclasses import dataclass

from finetune.config import DEFAULT_BASE_MODEL, DEFAULT_OUTPUT_DIR, DEFAULT_TRAIN_DATA

@dataclass
class LoraOptions:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    merge: bool = False  # also save a merged full model to <output>/merged
    target_modules: tuple | str | None = None  # None: from the model profile (model_profiles.yaml)


@dataclass
class TrainOptions:
    model: str = DEFAULT_BASE_MODEL
    data: str = DEFAULT_TRAIN_DATA
    output: str = DEFAULT_OUTPUT_DIR
    eval_ratio: float = 0.0
    epochs: float = 3
    lr: float | None = None  # default: 2e-5 full FT, 2e-4 LoRA
    batch_size: int = 4
    grad_accum: int = 4
    max_length: int = 1024
    warmup_ratio: float = 0.03
    seed: int = 42
    save_checkpoints: bool = False
    gradient_checkpointing: bool = True
    lora: LoraOptions | None = None  # None = full fine-tuning
    profile: str | None = None       # model profile name; None: resolved from `model`

    def __post_init__(self):
        if self.lr is None:
            self.lr = 2e-4 if self.lora else 2e-5
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if not 0 <= self.eval_ratio < 1:
            raise ValueError("eval_ratio must be in [0, 1)")
        if self.batch_size < 1 or self.grad_accum < 1:
            raise ValueError("batch_size and grad_accum must be >= 1")

    @property
    def method(self):
        return "lora" if self.lora else "full"

    @property
    def effective_batch_size(self):
        return self.batch_size * self.grad_accum
