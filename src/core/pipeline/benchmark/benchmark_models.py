"""Immutable domain models for Pipeline Benchmarking & Evaluation Framework (M5.4)."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator


class PipelineBenchmarkMetric(str, Enum):
    """Supported metric names for pipeline evaluation."""

    SUCCESS_RATE = "success_rate"
    MEAN_LATENCY_MS = "mean_latency_ms"
    P50_LATENCY_MS = "p50_latency_ms"
    P95_LATENCY_MS = "p95_latency_ms"
    P99_LATENCY_MS = "p99_latency_ms"
    THROUGHPUT_QPS = "throughput_qps"
    RETRY_RATE = "retry_rate"
    MEAN_RETRY_ATTEMPTS = "mean_retry_attempts"
    TIMEOUT_RATE = "timeout_rate"
    RECOVERY_RATE = "recovery_rate"
    DETERMINISM_RATE = "determinism_rate"


class PipelineBenchmarkItem(BaseModel):
    """Immutable single benchmark item binding a claim to expected success."""

    item_id: str = Field(..., min_length=1)
    claim: str = Field(..., min_length=1)
    pipeline_profile_id: str = Field(..., min_length=1)
    expected_success: bool = Field(default=True)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


class PipelineBenchmarkDataset(BaseModel):
    """Immutable collection of PipelineBenchmarkItem records."""

    dataset_id: str = Field(..., min_length=1)
    items: tuple[PipelineBenchmarkItem, ...] = Field(..., min_length=1)

    model_config = ConfigDict(frozen=True)


class PipelineBenchmarkSuite(BaseModel):
    """Immutable benchmark suite tying a dataset and configuration together."""

    suite_id: str = Field(..., min_length=1)
    dataset: PipelineBenchmarkDataset = Field(...)
    execution_config: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


class PipelineFailureRecord(BaseModel):
    """Immutable record of a pipeline execution failure."""

    exception_type: str = Field(..., min_length=1)
    error_message: str = Field(...)
    failure_category: str = Field(default="UNKNOWN")
    retry_attempts: int = Field(default=0, ge=0)

    model_config = ConfigDict(frozen=True)


class PipelineBenchmarkRawOutput(BaseModel):
    """Immutable collection of per-item raw execution records produced by the executor."""

    suite_id: str = Field(..., min_length=1)
    item_ids: tuple[str, ...] = Field(default_factory=tuple)
    claims: tuple[str, ...] = Field(default_factory=tuple)
    expected_successes: tuple[bool, ...] = Field(default_factory=tuple)
    actual_successes: tuple[bool, ...] = Field(default_factory=tuple)
    total_latencies_ms: tuple[float, ...] = Field(default_factory=tuple)
    stage_latencies_ms: tuple[dict[str, float], ...] = Field(default_factory=tuple)
    retry_attempt_counts: tuple[int, ...] = Field(default_factory=tuple)
    timeout_triggered: tuple[bool, ...] = Field(default_factory=tuple)
    recovery_invoked: tuple[bool, ...] = Field(default_factory=tuple)
    failures: tuple[PipelineFailureRecord | None, ...] = Field(default_factory=tuple)

    model_config = ConfigDict(frozen=True)


class PipelineBenchmarkMetrics(BaseModel):
    """Strongly typed immutable pipeline benchmark metrics."""

    success_rate: float = Field(..., ge=0.0, le=1.0)
    mean_latency_ms: float = Field(..., ge=0.0)
    p50_latency_ms: float = Field(..., ge=0.0)
    p95_latency_ms: float = Field(..., ge=0.0)
    p99_latency_ms: float = Field(..., ge=0.0)
    throughput_qps: float = Field(..., ge=0.0)
    retry_rate: float = Field(..., ge=0.0, le=1.0)
    mean_retry_attempts: float = Field(..., ge=0.0)
    timeout_rate: float = Field(..., ge=0.0, le=1.0)
    recovery_rate: float = Field(..., ge=0.0, le=1.0)
    determinism_rate: float = Field(..., ge=0.0, le=1.0)

    model_config = ConfigDict(frozen=True)


class PipelineStageBenchmarkMetrics(BaseModel):
    """Immutable per-stage latency breakdown aggregated across all benchmark items."""

    stage_id: str = Field(..., min_length=1)
    mean_latency_ms: float = Field(..., ge=0.0)
    p50_latency_ms: float = Field(..., ge=0.0)
    p95_latency_ms: float = Field(..., ge=0.0)
    p99_latency_ms: float = Field(..., ge=0.0)

    model_config = ConfigDict(frozen=True)


class PipelineBenchmarkResult(BaseModel):
    """Immutable result of a benchmark run containing aggregate metrics and stage breakdown."""

    suite_id: str = Field(..., min_length=1)
    metrics: PipelineBenchmarkMetrics = Field(...)
    stage_metrics: dict[str, PipelineStageBenchmarkMetrics] = Field(
        default_factory=dict
    )
    latency_stats: dict[str, float] = Field(default_factory=dict)
    item_count: int = Field(..., ge=0)
    success: bool = Field(default=True)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


class PipelineBenchmarkReport(BaseModel):
    """Immutable final benchmark report documenting run metadata and outcomes."""

    suite_id: str = Field(..., min_length=1)
    result: PipelineBenchmarkResult = Field(...)
    profile_id: str = Field(..., min_length=1)
    pipeline_profile_id: str = Field(..., min_length=1)
    timestamp: str = Field(..., min_length=1)
    schema_version: str = Field(default="1.0.0")

    model_config = ConfigDict(frozen=True)


class PipelineBenchmarkDefinition(BaseModel):
    """Immutable configuration defining which metrics to evaluate."""

    enabled_metrics: tuple[PipelineBenchmarkMetric, ...] = Field(
        default=(
            PipelineBenchmarkMetric.SUCCESS_RATE,
            PipelineBenchmarkMetric.MEAN_LATENCY_MS,
            PipelineBenchmarkMetric.P50_LATENCY_MS,
            PipelineBenchmarkMetric.P95_LATENCY_MS,
            PipelineBenchmarkMetric.P99_LATENCY_MS,
            PipelineBenchmarkMetric.THROUGHPUT_QPS,
            PipelineBenchmarkMetric.RETRY_RATE,
            PipelineBenchmarkMetric.MEAN_RETRY_ATTEMPTS,
            PipelineBenchmarkMetric.TIMEOUT_RATE,
            PipelineBenchmarkMetric.RECOVERY_RATE,
            PipelineBenchmarkMetric.DETERMINISM_RATE,
        )
    )
    include_stage_breakdown: bool = Field(default=True)

    model_config = ConfigDict(frozen=True)


class PipelineBenchmarkProfile(BaseModel):
    """Immutable profile binding a suite and metric configuration."""

    profile_id: str = Field(..., min_length=1)
    suite_id: str = Field(..., min_length=1)
    definition: PipelineBenchmarkDefinition = Field(...)

    model_config = ConfigDict(frozen=True)


class PipelineBenchmarkProfileRegistry(BaseModel):
    """O(1) registry providing pipeline benchmark profile resolution."""

    profiles: tuple[PipelineBenchmarkProfile, ...] = Field(..., min_length=1)

    _profile_index: dict[str, PipelineBenchmarkProfile] = PrivateAttr(
        default_factory=dict
    )

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def _build_and_validate_index(self) -> "PipelineBenchmarkProfileRegistry":
        from src.core.exceptions import DuplicatePipelineBenchmarkProfileError

        index: dict[str, PipelineBenchmarkProfile] = {}
        for p in self.profiles:
            profile_id = p.profile_id
            if profile_id in index:
                raise DuplicatePipelineBenchmarkProfileError(
                    f"Duplicate benchmark profile ID detected: {profile_id}"
                )
            index[profile_id] = p
        object.__setattr__(self, "_profile_index", index)
        return self

    def resolve(self, profile_id: str) -> PipelineBenchmarkProfile:
        from src.core.exceptions import PipelineBenchmarkProfileNotFoundError

        if profile_id not in self._profile_index:
            raise PipelineBenchmarkProfileNotFoundError(
                f"Pipeline benchmark profile not found: {profile_id}"
            )
        return self._profile_index[profile_id]

    def validate_compatibility(self, suite_id: str) -> None:
        """Validates that a profile is registered for the specified suite_id."""
        for p in self.profiles:
            if p.suite_id == suite_id:
                return
        from src.core.exceptions import PipelineBenchmarkProfileNotFoundError

        raise PipelineBenchmarkProfileNotFoundError(
            f"No benchmark profile configured for suite ID: {suite_id}"
        )
