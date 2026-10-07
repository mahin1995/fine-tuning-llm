"""Moving to another runtime. Both speak the same API, so clients only change their base URL.

vLLM:   reads Hugging Face / fine-tuned model directories directly (LoRA via --enable-lora).
        Template variables (e.g. Qwen3 enable_thinking=False) are sent per request by clients
        as `chat_template_kwargs`.
Ollama: needs a GGUF file + Modelfile per model (fine-tuned models must be converted first).
"""
from pathlib import Path

from model_serving.config import ModelEntry

VLLM_IMAGE = "vllm/vllm-openai:latest"


def vllm_command(entry: ModelEntry, *, port: int = 8001, gpu_memory_utilization: float = 0.85,
                 max_model_len: int = 8192, hf_cache: str = "~/.cache/huggingface",
                 outputs_dir: str = "./outputs") -> list[str]:
    """`docker run` argv for serving `entry` with vLLM on the same port and model name.
    Host paths are made absolute: the argv is printed shell-quoted, so `~` would not expand."""
    hf_cache = str(Path(hf_cache).expanduser().resolve())
    cmd = ["docker", "run", "--rm", "--gpus", "all", "--ipc=host", "-p", f"127.0.0.1:{port}:8000",
           "-v", f"{hf_cache}:/root/.cache/huggingface"]
    source = entry.source
    if entry.is_local:
        cmd += ["-v", f"{Path(outputs_dir).expanduser().resolve()}:/models/outputs:ro"]
    cmd += [VLLM_IMAGE, "--model", source, "--served-model-name", entry.name,
            "--max-model-len", str(max_model_len), "--gpu-memory-utilization", str(gpu_memory_utilization),
            "--enable-auto-tool-choice", "--tool-call-parser", "hermes"]
    if entry.revision:
        cmd += ["--revision", entry.revision]
    return cmd


OLLAMA_STEPS = """\
Ollama serves GGUF models. For a Hugging Face model with an official GGUF build:
    ollama pull <name>                      # e.g. qwen2.5:7b-instruct
For a model fine-tuned with this repo:
    1. convert:  python llama.cpp/convert_hf_to_gguf.py <model dir> --outfile model.gguf
    2. Modelfile with `FROM ./model.gguf` and the model's chat TEMPLATE
    3. ollama create <name> -f Modelfile
Then point clients at http://localhost:11434/v1 and run
    python -m model_serving check --base-url http://localhost:11434/v1 --model <name>
"""
