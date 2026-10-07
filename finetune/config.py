"""Settings shared by several packages. Keep this module free of heavy imports."""

DEFAULT_BASE_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_OUTPUT_DIR = "outputs/qwen3-ft"
DEFAULT_TRAIN_DATA = "data/train.jsonl"
DEFAULT_EVAL_DATA = "data/eval.jsonl"

# Chat template variables (e.g. Qwen3's enable_thinking=False) and LoRA target layers are
# per model family: see model_profiles.yaml and finetune.profiles.resolve_profile().
