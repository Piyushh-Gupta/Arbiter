"""Centralized configuration system using Pydantic Settings."""

from typing import Literal, cast

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.core.constants import APP_NAME, DEFAULT_ENV, DEFAULT_LOG_LEVEL
from src.core.paths import ProjectPaths


class DatabaseSettings(BaseModel):
    """Database configuration settings."""

    url: str = Field(default="sqlite:///./arbiter.db")


class LoggingSettings(BaseModel):
    """Logging configuration settings."""

    level: str = Field(default=DEFAULT_LOG_LEVEL)


class ActiveDatasetSettings(BaseModel):
    """Settings representing the currently active dataset."""

    id: str | None = Field(default=None)
    version: str | None = Field(default=None)


class DownloadSettings(BaseModel):
    """Download retry and timeout configurations."""

    max_retries: int = Field(default=3, ge=0)
    backoff_factor: float = Field(default=2.0, ge=1.0)
    timeout_seconds: float = Field(default=30.0, gt=0.0)


class Settings(BaseSettings):
    """Root application settings."""

    app_name: str = Field(default=APP_NAME)
    environment: Literal["development", "production", "test"] = Field(
        default=cast(Literal["development", "production", "test"], DEFAULT_ENV)
    )

    # Sub-configurations
    dataset: ActiveDatasetSettings = ActiveDatasetSettings()
    download: DownloadSettings = DownloadSettings()
    db: DatabaseSettings = DatabaseSettings()
    log: LoggingSettings = LoggingSettings()
    nli: "NLIVerifierSettings" = Field(default_factory=lambda: NLIVerifierSettings())

    # Expose paths through config for unified access
    paths: type[ProjectPaths] = ProjectPaths

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
    )


class NLIVerifierSettings(BaseModel):
    """NLI verifier configuration settings."""

    model_id: str = Field(default="cross-encoder/nli-distilroberta-base")
    tokenizer_id: str = Field(default="cross-encoder/nli-distilroberta-base")
    device: str = Field(default="cpu")
    precision: str = Field(default="fp32")
    max_sequence_length: int = Field(default=512)
    batch_size: int = Field(default=8)


settings = Settings()
