"""Centralized configuration system using Pydantic Settings."""

from typing import Any, Literal, cast

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


class APIServiceSettings(BaseModel):
    """Settings for the API Service Layer."""

    active_profile_id: str = Field(default="default_api_service")
    require_correlation_id: bool = Field(default=True)
    timeout_seconds: float = Field(default=30.0)


class ApiContractSettings(BaseModel):
    """Settings for the API Contract Layer."""

    active_profile_id: str = Field(default="default_api_contract")
    require_correlation_id: bool = Field(default=True)
    strict_validation: bool = Field(default=True)


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
    aggregation: "AggregationSettings" = Field(
        default_factory=lambda: AggregationSettings()
    )
    calibration: "CalibrationSettings" = Field(
        default_factory=lambda: CalibrationSettings()
    )
    benchmark: "BenchmarkSettings" = Field(default_factory=lambda: BenchmarkSettings())
    explainability: "ExplainabilitySettings" = Field(
        default_factory=lambda: ExplainabilitySettings()
    )
    verification_optimization: "VerificationOptimizationSettings" = Field(
        default_factory=lambda: VerificationOptimizationSettings()
    )
    verification_hardening: "VerificationHardeningSettings" = Field(
        default_factory=lambda: VerificationHardeningSettings()
    )
    pipeline_telemetry: "PipelineTelemetrySettings" = Field(
        default_factory=lambda: PipelineTelemetrySettings()
    )
    pipeline_resilience: "PipelineResilienceSettings" = Field(
        default_factory=lambda: PipelineResilienceSettings()
    )
    pipeline_benchmark: "PipelineBenchmarkSettings" = Field(
        default_factory=lambda: PipelineBenchmarkSettings()
    )
    pipeline_explanation: "PipelineExplanationSettings" = Field(
        default_factory=lambda: PipelineExplanationSettings()
    )
    pipeline_operations: "PipelineOperationsSettings" = Field(
        default_factory=lambda: PipelineOperationsSettings()
    )
    api_contracts: "ApiContractSettings" = Field(
        default_factory=lambda: ApiContractSettings()
    )
    api_services: "APIServiceSettings" = Field(
        default_factory=lambda: APIServiceSettings()
    )

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


class AggregationSettings(BaseModel):
    """Configuration settings for multi-evidence aggregation."""

    consensus_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    contradiction_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    default_strategy: str = Field(default="MAX_CONFIDENCE")
    default_weigher: str = Field(default="default")


class CalibrationSettings(BaseModel):
    """Configuration settings for confidence calibration."""

    default_strategy: str = Field(default="IDENTITY")
    temperature: float = Field(default=1.5, gt=0.0)
    platt_slope: float = Field(default=1.0)
    platt_intercept: float = Field(default=0.0)


class BenchmarkSettings(BaseModel):
    """Configuration settings for offline benchmarking."""

    default_profile: str = Field(default="default_benchmark")
    dataset_base_path: str = Field(default="data/benchmark")


class ExplainabilitySettings(BaseModel):
    """Configuration settings for explainability."""

    default_profile: str = Field(default="composite_explanation")


class VerificationOptimizationSettings(BaseModel):
    """Configuration settings for verification production optimization."""

    default_profile: str = Field(default="default_optimization")


class VerificationHardeningSettings(BaseModel):
    """Configuration settings for verification production hardening."""

    operational_profile: str = Field(default="default_operational")
    logging_configuration: dict[str, Any] = Field(default_factory=dict)
    readiness_configuration: dict[str, Any] = Field(default_factory=dict)
    telemetry_configuration: dict[str, Any] = Field(default_factory=dict)


class PipelineTelemetrySettings(BaseModel):
    """Configuration settings for pipeline telemetry."""

    enabled: bool = Field(default=True)
    snapshot_on_every_execution: bool = Field(default=False)
    active_exporters: list[str] = Field(
        default_factory=lambda: ["default_log_exporter"]
    )
    log_level: str = Field(default="INFO")
    include_stage_breakdown: bool = Field(default=True)
    json_output_path: str = Field(default="data/telemetry/snapshot.json")
    json_pretty_print: bool = Field(default=False)


class PipelineResilienceSettings(BaseModel):
    """Configuration settings for pipeline resilience."""

    enabled: bool = Field(default=True)
    max_retry_attempts: int = Field(default=3, ge=1, le=10)
    retry_delay_ms: float = Field(default=100.0, ge=0.0)
    retryable_exceptions: list[str] = Field(
        default_factory=lambda: ["PipelineStageExecutionError"]
    )
    timeout_enabled: bool = Field(default=True)
    timeout_ms: float = Field(default=30_000.0, gt=0.0)
    active_resilience_profile_id: str = Field(default="default_resilience")
    recovery_strategy_id: str = Field(default="default_recovery")


class PipelineBenchmarkSettings(BaseModel):
    """Configuration settings for pipeline benchmarking."""

    enabled: bool = Field(default=True)
    active_profile_id: str = Field(default="default_pipeline_benchmark")
    default_suite_id: str = Field(default="default_pipeline_suite")
    enabled_metrics: list[str] = Field(
        default_factory=lambda: [
            "success_rate",
            "mean_latency_ms",
            "p50_latency_ms",
            "p95_latency_ms",
            "p99_latency_ms",
            "throughput_qps",
            "retry_rate",
            "mean_retry_attempts",
            "timeout_rate",
            "recovery_rate",
            "determinism_rate",
        ]
    )
    include_stage_breakdown: bool = Field(default=True)


class PipelineExplanationSettings(BaseModel):
    """Configuration settings for pipeline explainability."""

    enabled: bool = Field(default=True)
    active_profile_id: str = Field(default="default_pipeline_explanation")
    default_strategy_id: str = Field(default="pipeline_composite")
    default_renderer_id: str = Field(default="markdown")
    include_stage_breakdown: bool = Field(default=True)
    include_resilience_trace: bool = Field(default=True)
    include_telemetry_summary: bool = Field(default=True)
    include_configuration_fingerprint: bool = Field(default=True)


class PipelineOperationsSettings(BaseModel):
    """Configuration settings for pipeline operations."""

    enabled: bool = Field(default=True)
    active_profile_id: str = Field(default="default_pipeline_operations")


# Modify Settings class to include pipeline subsystems
settings = Settings()
