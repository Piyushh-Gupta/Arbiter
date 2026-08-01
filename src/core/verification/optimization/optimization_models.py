"""Immutable domain models for Verification Production Optimization (M2.8)."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator


class TelemetryLevel(str, Enum):
    """Observability verbosity levels for telemetry logging."""

    NONE = "none"
    BASIC = "basic"
    DETAILED = "detailed"


class OptimizationMode(str, Enum):
    """Pre-set operational tuning targets."""

    LATENCY = "latency"
    THROUGHPUT = "throughput"
    BALANCED = "balanced"


class VerificationOptimizationDefinition(BaseModel):
    """Immutable configuration tuning parameters for verification pipeline optimization."""

    verifier_batch_size: int = Field(default=16, gt=0)
    aggregation_batch_size: int = Field(default=16, gt=0)
    calibration_batch_size: int = Field(default=16, gt=0)
    explanation_batch_size: int = Field(default=16, gt=0)
    max_concurrent_requests: int = Field(default=4, gt=0)
    request_timeout_ms: float = Field(default=5000.0, gt=0.0)
    prefetch_size: int = Field(default=32, gt=0)
    telemetry_enabled: bool = Field(default=True)
    profiling_enabled: bool = Field(default=False)
    telemetry_level: TelemetryLevel = Field(default=TelemetryLevel.BASIC)
    optimization_mode: OptimizationMode = Field(default=OptimizationMode.BALANCED)

    model_config = ConfigDict(frozen=True, use_enum_values=True)


class VerificationExecutionMetrics(BaseModel):
    """Immutable snapshot metrics captured for a single pipeline request execution."""

    verification_latency_ms: float = Field(..., ge=0.0)
    aggregation_latency_ms: float = Field(..., ge=0.0)
    calibration_latency_ms: float = Field(..., ge=0.0)
    explanation_latency_ms: float = Field(..., ge=0.0)
    total_latency_ms: float = Field(..., ge=0.0)
    throughput_qps: float = Field(..., ge=0.0)
    memory_usage_bytes: int = Field(..., ge=0)
    batch_sizes: dict[str, int] = Field(..., min_length=1)
    concurrency_active_requests: int = Field(..., ge=0)

    model_config = ConfigDict(frozen=True)


class VerificationTelemetrySnapshot(BaseModel):
    """Aggregated thread-safe operational metrics collected across multiple requests."""

    total_requests: int = Field(..., ge=0)
    average_latency_ms: float = Field(..., ge=0.0)
    p95_latency_ms: float = Field(..., ge=0.0)
    throughput_qps: float = Field(..., ge=0.0)
    peak_concurrency: int = Field(..., ge=0)

    model_config = ConfigDict(frozen=True)


class VerificationOptimizationTrace(BaseModel):
    """Audit trace documenting optimization decisions and environment state per run."""

    profile_id: str = Field(..., min_length=1)
    semaphore_active_slots: int = Field(..., ge=0)
    timeout_ms: float = Field(..., gt=0.0)
    batch_configuration: dict[str, int] = Field(..., min_length=1)
    telemetry_configured: bool = Field(...)
    execution_timestamp: str = Field(..., min_length=1)

    model_config = ConfigDict(frozen=True)


class VerificationOptimizationProfile(BaseModel):
    """Pairing of an optimization identifier and its immutable definition."""

    profile_id: str = Field(..., min_length=1)
    definition: VerificationOptimizationDefinition = Field(...)

    model_config = ConfigDict(frozen=True)


class VerificationOptimizationProfileRegistry(BaseModel):
    """O(1) registry mapping identifiers to unique VerificationOptimizationProfiles."""

    profiles: tuple[VerificationOptimizationProfile, ...] = Field(..., min_length=1)

    _profile_index: dict[str, VerificationOptimizationProfile] = PrivateAttr(
        default_factory=dict
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _build_and_validate_index(self) -> "VerificationOptimizationProfileRegistry":
        from src.core.exceptions import DuplicateOptimizationProfileError

        index: dict[str, VerificationOptimizationProfile] = {}
        for p in self.profiles:
            if p.profile_id in index:
                raise DuplicateOptimizationProfileError(
                    f"Duplicate verification optimization profile identifier: {p.profile_id}"
                )
            index[p.profile_id] = p
        object.__setattr__(self, "_profile_index", index)
        return self

    def resolve(self, profile_id: str) -> VerificationOptimizationProfile:
        from src.core.exceptions import OptimizationProfileNotFoundError

        if profile_id not in self._profile_index:
            raise OptimizationProfileNotFoundError(
                f"Verification optimization profile not found: {profile_id}"
            )
        return self._profile_index[profile_id]
