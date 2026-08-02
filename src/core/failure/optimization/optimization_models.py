"""Immutable domain models for Production Failure Analysis Optimization (M3.8)."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator


class FailureOptimizationDefinition(BaseModel):
    """Immutable configuration for failure analysis production optimization."""

    batch_size: int = Field(default=16, gt=0)
    max_concurrent_requests: int = Field(default=4, gt=0)
    timeout_ms: float = Field(default=5000.0, gt=0.0)
    telemetry_enabled: bool = Field(default=True)
    profiling_enabled: bool = Field(default=False)
    memory_limit_mb: float | None = Field(default=None)
    cache_profile_id: str | None = Field(default=None)

    model_config = ConfigDict(frozen=True)


class FailureExecutionMetrics(BaseModel):
    """Immutable execution metrics captured for a single failure analysis run."""

    analysis_latency_ms: float = Field(default=0.0, ge=0.0)
    correlation_latency_ms: float = Field(default=0.0, ge=0.0)
    attribution_latency_ms: float = Field(default=0.0, ge=0.0)
    severity_latency_ms: float = Field(default=0.0, ge=0.0)
    explainability_latency_ms: float = Field(default=0.0, ge=0.0)
    total_latency_ms: float = Field(..., ge=0.0)
    memory_usage_mb: float = Field(default=0.0, ge=0.0)
    analyzer_count: int = Field(default=1, ge=0)

    model_config = ConfigDict(frozen=True)


class FailureTelemetryRecord(BaseModel):
    """Immutable operational metrics record emitted after execution."""

    request_id: str = Field(..., min_length=1)
    timestamp: str = Field(..., min_length=1)
    execution_metrics: FailureExecutionMetrics = Field(...)
    success: bool = Field(default=True)
    error_message: str | None = Field(default=None)

    model_config = ConfigDict(frozen=True)


class FailureTelemetrySnapshot(BaseModel):
    """Immutable aggregated operational metrics snapshot."""

    total_requests: int = Field(default=0, ge=0)
    average_latency_ms: float = Field(default=0.0, ge=0.0)
    p95_latency_ms: float = Field(default=0.0, ge=0.0)
    p99_latency_ms: float = Field(default=0.0, ge=0.0)
    throughput_qps: float = Field(default=0.0, ge=0.0)
    failure_count: int = Field(default=0, ge=0)

    model_config = ConfigDict(frozen=True)


class FailureOperationalProfile(BaseModel):
    """Immutable operational profile binding optimization, timeout, telemetry, and concurrency policies."""

    profile_id: str = Field(..., min_length=1)
    optimization_definition: FailureOptimizationDefinition = Field(...)
    timeout_policy: float = Field(default=5000.0, gt=0.0)
    telemetry_configuration: dict[str, Any] = Field(default_factory=dict)
    concurrency_policy: int = Field(default=4, gt=0)

    model_config = ConfigDict(frozen=True)


class FailureOptimizationProfile(BaseModel):
    """Immutable pairing of profile_id with optimization definition and controller."""

    profile_id: str = Field(..., min_length=1)
    definition: FailureOptimizationDefinition = Field(...)
    controller: Any = Field(...)
    operational_profile: FailureOperationalProfile | None = Field(default=None)

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _validate_compatibility(self) -> "FailureOptimizationProfile":
        if hasattr(self.controller, "validate_compatibility"):
            self.controller.validate_compatibility(self.definition)
        return self


class FailureOptimizationProfileRegistry(BaseModel):
    """O(1) registry resolver for failure optimization profiles."""

    profiles: tuple[FailureOptimizationProfile, ...] = Field(..., min_length=1)

    _profile_index: dict[str, FailureOptimizationProfile] = PrivateAttr(
        default_factory=dict
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _build_and_validate_index(self) -> "FailureOptimizationProfileRegistry":
        from src.core.exceptions import DuplicateFailureAnalysisProfileError

        index: dict[str, FailureOptimizationProfile] = {}
        for p in self.profiles:
            if p.profile_id in index:
                raise DuplicateFailureAnalysisProfileError(
                    f"Duplicate failure optimization profile identifier: {p.profile_id}"
                )
            index[p.profile_id] = p
        object.__setattr__(self, "_profile_index", index)
        return self

    def resolve(self, profile_id: str) -> FailureOptimizationProfile:
        from src.core.exceptions import FailureAnalysisProfileNotFoundError

        if profile_id not in self._profile_index:
            raise FailureAnalysisProfileNotFoundError(
                f"Failure optimization profile not found: {profile_id}"
            )
        return self._profile_index[profile_id]
