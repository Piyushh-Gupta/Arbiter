"""Immutable domain models for pipeline resilience."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from src.core.pipeline.pipeline_models import PipelineExecutionResult


class RetryDefinition(BaseModel):
    """Immutable configuration for a deterministic retry sequence."""

    max_attempts: int = Field(default=3, ge=1, le=10)
    retry_delay_ms: float = Field(default=100.0, ge=0.0)
    retryable_on: tuple[str, ...] = Field(default=("PipelineStageExecutionError",))
    model_config = ConfigDict(frozen=True)


class TimeoutDefinition(BaseModel):
    """Immutable timeout configuration for a single pipeline execution."""

    enabled: bool = Field(default=True)
    timeout_ms: float = Field(default=30000.0, gt=0.0)
    model_config = ConfigDict(frozen=True)


class RecoveryDefinition(BaseModel):
    """Immutable configuration binding a named recovery strategy."""

    strategy_id: str = Field(..., min_length=1)
    enabled: bool = Field(default=True)
    model_config = ConfigDict(frozen=True)


class PipelineResilienceDefinition(BaseModel):
    """Top-level immutable resilience configuration for a pipeline profile."""

    enabled: bool = Field(default=True)
    retry: RetryDefinition = Field(default_factory=RetryDefinition)
    timeout: TimeoutDefinition = Field(default_factory=TimeoutDefinition)
    recovery: RecoveryDefinition = Field(
        default_factory=lambda: RecoveryDefinition(strategy_id="default_recovery")
    )
    model_config = ConfigDict(frozen=True)


class RetryAttemptRecord(BaseModel):
    """Immutable record of a single retry attempt."""

    attempt_number: int = Field(..., ge=1)
    error_type: str = Field(..., min_length=1)
    error_message: str = Field(...)
    latency_ms: float = Field(..., ge=0.0)
    model_config = ConfigDict(frozen=True)


class RetryExecutionTrace(BaseModel):
    """Immutable audit trace of the complete retry sequence for one execution."""

    execution_id: str = Field(..., min_length=1)
    total_attempts: int = Field(..., ge=1)
    succeeded: bool = Field(...)
    attempts: tuple[RetryAttemptRecord, ...] = Field(...)
    total_retry_overhead_ms: float = Field(..., ge=0.0)
    terminal_error: str | None = Field(default=None)
    model_config = ConfigDict(frozen=True)


class ResilienceRuntimeMetadata(BaseModel):
    """Immutable metadata attached to every resilience-wrapped execution."""

    pipeline_profile_id: str = Field(..., min_length=1)
    resilience_profile_id: str = Field(..., min_length=1)
    timeout_enforced: bool = Field(...)
    retry_trace: RetryExecutionTrace = Field(...)
    recovery_invoked: bool = Field(default=False)
    recovery_strategy_id: str | None = Field(default=None)
    observed_at: datetime = Field(...)
    schema_version: str = Field(default="1.0.0")
    model_config = ConfigDict(frozen=True)


class PipelineRecoveryResult(BaseModel):
    """Immutable result produced when a recovery strategy is invoked."""

    recovery_strategy_id: str = Field(..., min_length=1)
    succeeded: bool = Field(...)
    result: PipelineExecutionResult | None = Field(default=None)
    failure_reason: str | None = Field(default=None)
    resilience_metadata: ResilienceRuntimeMetadata = Field(...)
    model_config = ConfigDict(frozen=True)
