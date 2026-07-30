"""Immutable domain models for Production Retrieval Optimization."""

import typing

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

if typing.TYPE_CHECKING:
    from src.core.retrieval.optimization.concurrency import ConcurrencyLimiter
else:
    ConcurrencyLimiter = typing.Any

__all__ = [
    "ExecutionPolicy",
    "OptimizationDefinition",
    "OptimizationProfile",
    "OptimizationProfileRegistry",
    "RetrievalExecutionMetrics",
    "TelemetrySnapshot",
]


class ExecutionPolicy(BaseModel):
    """Immutable execution policy encapsulating batching, concurrency, timeout, and prefetch limits."""

    retrieval_batch_size: int = Field(
        default=16, gt=0, description="Batch size for stage-1 candidate retrieval."
    )
    reranking_batch_size: int = Field(
        default=8, gt=0, description="Batch size for stage-2 reranking model scoring."
    )
    max_concurrent_requests: int = Field(
        default=4, gt=0, description="Maximum concurrent retrieval executions."
    )
    request_timeout_ms: float = Field(
        default=5000.0, gt=0.0, description="Maximum allowed latency in milliseconds."
    )
    document_prefetch_size: int = Field(
        default=32, gt=0, description="Maximum chunks pre-loaded per lookup."
    )
    cache_profile_id: str | None = Field(
        default=None, description="Optional associated cache profile ID."
    )

    model_config = ConfigDict(frozen=True)


class RetrievalExecutionMetrics(BaseModel):
    """Immutable per-request execution metrics captured via monotonic timers."""

    retrieval_latency_ms: float = Field(..., ge=0.0)
    reranking_latency_ms: float = Field(..., ge=0.0)
    cache_latency_ms: float = Field(..., ge=0.0)
    document_lookup_latency_ms: float = Field(..., ge=0.0)
    total_latency_ms: float = Field(..., ge=0.0)
    candidate_count: int = Field(..., ge=0)
    passage_count: int = Field(..., ge=0)
    cache_hit: bool = Field(...)

    model_config = ConfigDict(frozen=True)


class TelemetrySnapshot(BaseModel):
    """Immutable snapshot of aggregated operational telemetry."""

    total_requests: int = Field(..., ge=0)
    cache_hits: int = Field(..., ge=0)
    cache_misses: int = Field(..., ge=0)
    average_latency_ms: float = Field(..., ge=0.0)
    p95_latency_ms: float = Field(..., ge=0.0)
    throughput_qps: float = Field(..., ge=0.0)

    model_config = ConfigDict(frozen=True)


class OptimizationDefinition(BaseModel):
    """Immutable configuration for production retrieval optimization."""

    execution_policy: ExecutionPolicy = Field(
        default_factory=ExecutionPolicy,
        description="Immutable execution parameters policy.",
    )
    telemetry_enabled: bool = Field(
        default=True, description="Whether operational telemetry is enabled."
    )
    profiling_enabled: bool = Field(
        default=False, description="Whether detailed execution profiling is enabled."
    )

    model_config = ConfigDict(frozen=True)


class OptimizationProfile(BaseModel):
    """Immutable reusable profile binding an optimization definition and execution policy."""

    profile_id: str = Field(
        ..., description="Unique identifier for this optimization profile."
    )
    definition: OptimizationDefinition = Field(
        ..., description="Optimization definition parameters."
    )
    execution_policy: ExecutionPolicy = Field(
        ..., description="Execution policy parameters."
    )
    concurrency_limiter: ConcurrencyLimiter = Field(
        ..., description="Injected concurrency limiter instance."
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class OptimizationProfileRegistry(BaseModel):
    """Immutable O(1) registry for resolving named optimization profiles."""

    profiles: tuple[OptimizationProfile, ...] = Field(
        ..., min_length=1, description="Collection of registered optimization profiles."
    )

    _profile_index: dict[str, OptimizationProfile] = PrivateAttr(default_factory=dict)

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _build_and_validate_index(self) -> "OptimizationProfileRegistry":
        from src.core.exceptions import DuplicateOptimizationProfileError

        index: dict[str, OptimizationProfile] = {}
        for profile in self.profiles:
            if profile.profile_id in index:
                raise DuplicateOptimizationProfileError(
                    f"Duplicate optimization profile identifier: {profile.profile_id}"
                )
            index[profile.profile_id] = profile

        object.__setattr__(self, "_profile_index", index)
        return self

    def resolve(self, profile_id: str) -> OptimizationProfile:
        from src.core.exceptions import OptimizationProfileNotFoundError

        if profile_id not in self._profile_index:
            raise OptimizationProfileNotFoundError(
                f"Optimization profile not found: {profile_id}"
            )
        return self._profile_index[profile_id]
