"""Immutable domain models for pipeline telemetry."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class PipelineStageTelemetryRecord(BaseModel):
    """Immutable telemetry record for a single stage execution."""

    stage_id: str = Field(..., min_length=1)
    profile_id: str = Field(..., min_length=1)
    latency_ms: float = Field(..., ge=0.0)
    success: bool = Field(...)
    model_config = ConfigDict(frozen=True)


class PipelineTelemetryEvent(BaseModel):
    """Immutable telemetry event representing a single pipeline execution observation."""

    execution_id: str = Field(..., min_length=1)
    pipeline_id: str = Field(..., min_length=1)
    claim_length: int = Field(..., ge=0)
    total_latency_ms: float = Field(..., ge=0.0)
    success: bool = Field(...)
    stage_records: tuple[PipelineStageTelemetryRecord, ...] = Field(...)
    configuration_fingerprint: str = Field(...)
    execution_environment: str = Field(...)
    observed_at: datetime = Field(...)
    schema_version: str = Field(default="1.0.0")
    model_config = ConfigDict(frozen=True)


class PipelineStageAggregation(BaseModel):
    """Immutable per-stage aggregation across multiple executions."""

    stage_id: str = Field(..., min_length=1)
    profile_id: str = Field(..., min_length=1)
    execution_count: int = Field(..., ge=0)
    success_count: int = Field(..., ge=0)
    failure_count: int = Field(..., ge=0)
    mean_latency_ms: float = Field(..., ge=0.0)
    p50_latency_ms: float = Field(..., ge=0.0)
    p90_latency_ms: float = Field(..., ge=0.0)
    p99_latency_ms: float = Field(..., ge=0.0)
    success_rate: float = Field(..., ge=0.0, le=1.0)
    model_config = ConfigDict(frozen=True)


class PipelineTelemetrySnapshot(BaseModel):
    """Immutable aggregated snapshot of telemetry across all observed executions."""

    pipeline_id: str = Field(..., min_length=1)
    total_executions: int = Field(..., ge=0)
    successful_executions: int = Field(..., ge=0)
    failed_executions: int = Field(..., ge=0)
    mean_total_latency_ms: float = Field(..., ge=0.0)
    p50_total_latency_ms: float = Field(..., ge=0.0)
    p90_total_latency_ms: float = Field(..., ge=0.0)
    p99_total_latency_ms: float = Field(..., ge=0.0)
    overall_success_rate: float = Field(..., ge=0.0, le=1.0)
    stage_aggregations: tuple[PipelineStageAggregation, ...] = Field(...)
    snapshot_timestamp: datetime = Field(...)
    schema_version: str = Field(default="1.0.0")
    model_config = ConfigDict(frozen=True)


class PipelineTelemetryReport(BaseModel):
    """Immutable telemetry report produced by an exporter on demand."""

    snapshot: PipelineTelemetrySnapshot = Field(...)
    format: str = Field(..., min_length=1)
    content: str = Field(...)
    generated_at: datetime = Field(...)
    model_config = ConfigDict(frozen=True)
