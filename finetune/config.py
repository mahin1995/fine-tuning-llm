"""Settings shared by several packages. Keep this module free of heavy imports."""

DEFAULT_BASE_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_OUTPUT_DIR = "outputs/qwen3-ft"
DEFAULT_TRAIN_DATA = "data/train.jsonl"
DEFAULT_EVAL_DATA = "data/eval.jsonl"

# Qwen3 renders an empty <think></think> block when thinking is disabled. Training
# and inference must use the same setting, otherwise the prompt format differs.
# Every apply_chat_template call must pass these kwargs; never duplicate them.
CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}
