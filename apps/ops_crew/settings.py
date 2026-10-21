"""Runtime configuration from environment variables (prefix OPS_). No secrets live here:
API keys are read by the LLM factory from the env var each LLM profile names."""
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_DIR = Path(__file__).parent


class OpsSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OPS_", extra="ignore")

    train_data: Path = Path("data/train.jsonl")
    eval_data: Path = Path("data/eval.jsonl")
    state_dir: Path = Path("outputs/ops")          # audit log + idempotency store
    config_dir: Path = PACKAGE_DIR / "config"      # agents.yaml, tasks.yaml, llms.yaml

    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_retries: int = Field(default=2, ge=0, le=5)          # crew retries after the first attempt
    crew_timeout_seconds: float = Field(default=300.0, gt=0)  # whole crew, per attempt
    max_request_chars: int = Field(default=4000, gt=0)

    llm_profile: str | None = None  # override every agent's LLM profile (see config/llms.yaml)
    refine: bool = True  # self-correction + reflection via the refiner package (config/refine.yaml)

    @property
    def audit_log_path(self) -> Path:
        return self.state_dir / "audit.jsonl"

    @property
    def idempotency_path(self) -> Path:
        return self.state_dir / "idempotency.jsonl"
