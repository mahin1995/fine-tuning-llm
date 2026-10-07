"""Fetch model files from the Hugging Face Hub. Needs only huggingface_hub (no torch)."""
from pathlib import Path

from model_serving.config import ModelEntry

# Weights, tokenizer and chat template; skips other checkpoint formats (.bin, .pt, .gguf, ...).
ALLOW_PATTERNS = ["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "*.tiktoken", "tokenizer*"]


class DownloadError(RuntimeError):
    pass


def fetch(entry: ModelEntry, cache_dir: str | None = None) -> Path:
    """Return a local directory with the model. Local sources are only checked, never copied."""
    if entry.is_local:
        path = Path(entry.source)
        if not (path / "config.json").is_file() and not (path / "adapter_config.json").is_file():
            raise DownloadError(f"{path} is not a model directory (no config.json / adapter_config.json)")
        return path
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(repo_id=entry.source, revision=entry.revision, cache_dir=cache_dir,
                                      allow_patterns=ALLOW_PATTERNS))
    except Exception as e:  # network, auth (gated model), unknown repo
        raise DownloadError(f"could not download {entry.source}: {type(e).__name__}: {e}") from e


def size_on_disk(path: Path) -> int:
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
