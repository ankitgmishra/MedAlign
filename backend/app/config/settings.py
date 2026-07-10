"""
Centralised application settings powered by pydantic-settings.

All configuration is loaded from environment variables / .env files.
Nothing is hardcoded.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Resolve the project root (.env lives next to `backend/`) ────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


class ModelSettings(BaseSettings):
    """Settings specific to model loading and inference."""

    model_config = SettingsConfigDict(
        env_prefix="MODEL_",
        env_file=str(_PROJECT_ROOT / ".env"),
        extra="ignore",
    )

    default_base_model: str = Field(
        default="Qwen/Qwen2.5-0.5B",
        description="HuggingFace model ID for the base model.",
    )
    default_sft_path: Optional[str] = Field(
        default=None,
        description="Path to the SFT adapter checkpoint.",
    )
    max_new_tokens: int = Field(default=20, ge=1)
    default_seed: int = Field(default=42)
    load_in_4bit: bool = Field(default=True)
    quantization_type: str = Field(default="nf4")
    compute_dtype: str = Field(default="bfloat16")
    use_double_quant: bool = Field(default=True)


class InferenceSettings(BaseSettings):
    """Settings for external LLM inference providers."""

    model_config = SettingsConfigDict(
        env_prefix="INFERENCE_",
        env_file=str(_PROJECT_ROOT / ".env"),
        extra="ignore",
    )

    ollama_base_url: str = Field(default="http://localhost:11434")
    fallback_models: str = Field(
        default="llama3.2:latest,llama3.2:latest",
        description="Comma-separated list of Ollama models to try in sequence if one fails."
    )
    max_retries: int = Field(default=3, ge=1)

    @property
    def fallback_models_list(self) -> list[str]:
        return [m.strip() for m in self.fallback_models.split(",") if m.strip()]


class StorageSettings(BaseSettings):
    """Settings for file I/O paths."""

    model_config = SettingsConfigDict(
        env_prefix="STORAGE_",
        env_file=str(_PROJECT_ROOT / ".env"),
        extra="ignore",
    )

    datasets_dir: Path = Field(default=_PROJECT_ROOT / "datasets")
    outputs_dir: Path = Field(default=_PROJECT_ROOT / "outputs")
    models_dir: Path = Field(default=_PROJECT_ROOT / "models")


class Settings(BaseSettings):
    """Top-level application settings aggregating all sub-settings."""

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────────────
    app_name: str = Field(default="MedAlign")
    app_version: str = Field(default="0.1.0")
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
    )
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")

    # ── Sub-settings (composed, not inherited) ───────────────────────────────
    model: ModelSettings = Field(default_factory=ModelSettings)
    inference: InferenceSettings = Field(default_factory=InferenceSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton of application settings."""
    return Settings()
