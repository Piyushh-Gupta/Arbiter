"""Immutable domain models for the Failure Analysis Benchmarking subsystem (M3.6)."""

import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from src.core.failure.failure_models import (
    FailureCategory,
    FailureRootCause,
    FailureSeverity,
)


class FailureBenchmarkItem(BaseModel):
    """One sample in a failure benchmark dataset."""

    item_id: str = Field(..., min_length=1)
    analyzer_execution_results: tuple[Any, ...] = Field(default_factory=tuple)
    expected_category: FailureCategory = Field(...)
    expected_root_cause: FailureRootCause = Field(...)
    expected_severity: FailureSeverity = Field(...)

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class FailureBenchmarkDefinition(BaseModel):
    """Immutable configuration for benchmark execution."""

    enabled_metrics: tuple[str, ...] = Field(default_factory=tuple)
    max_latency_ms: float = Field(default=5000.0, gt=0.0)
    determinism_runs: int = Field(default=3, ge=1)
    dataset_version: str = Field(default="1.0")

    model_config = ConfigDict(frozen=True)


class FailureBenchmarkSuite(BaseModel):
    """Immutable encapsulation of a benchmark dataset and its execution profile."""

    suite_id: str = Field(..., min_length=1)
    dataset: Any = Field(...)
    enabled_metrics: tuple[str, ...] = Field(default_factory=tuple)
    evaluation_profile: str = Field(default="default")
    execution_parameters: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class FailureBenchmarkResult(BaseModel):
    """Immutable per-metric benchmark result collection."""

    metric_values: dict[str, float] = Field(default_factory=dict)
    confusion_statistics: dict[str, Any] = Field(default_factory=dict)
    latency_statistics: dict[str, float] = Field(default_factory=dict)
    robustness_statistics: dict[str, float] = Field(default_factory=dict)
    execution_metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


class FailureBenchmarkReport(BaseModel):
    """Immutable top-level benchmark report carrying results, provenance, and trace."""

    result: FailureBenchmarkResult = Field(...)
    configuration_fingerprint: str = Field(..., min_length=1)
    execution_timestamp: str = Field(..., min_length=1)
    benchmark_trace: tuple[str, ...] = Field(default_factory=tuple)

    model_config = ConfigDict(frozen=True)


class FailureBenchmarkProfile(BaseModel):
    """Immutable pairing of a profile identifier with a benchmark definition and runner."""

    profile_id: str = Field(..., min_length=1)
    definition: FailureBenchmarkDefinition = Field(...)
    runner: Any = Field(...)

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _validate_compatibility(self) -> "FailureBenchmarkProfile":
        if hasattr(self.runner, "validate_compatibility"):
            self.runner.validate_compatibility(self.definition)
        return self


class FailureBenchmarkProfileRegistry(BaseModel):
    """O(1) registry resolver for failure benchmark profiles."""

    profiles: tuple[FailureBenchmarkProfile, ...] = Field(..., min_length=1)

    _profile_index: dict[str, FailureBenchmarkProfile] = PrivateAttr(
        default_factory=dict
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _build_and_validate_index(self) -> "FailureBenchmarkProfileRegistry":
        from src.core.exceptions import DuplicateFailureAnalysisProfileError

        index: dict[str, FailureBenchmarkProfile] = {}
        for p in self.profiles:
            if p.profile_id in index:
                raise DuplicateFailureAnalysisProfileError(
                    f"Duplicate failure benchmark profile identifier: {p.profile_id}"
                )
            index[p.profile_id] = p
        object.__setattr__(self, "_profile_index", index)
        return self

    def resolve(self, profile_id: str) -> FailureBenchmarkProfile:
        from src.core.exceptions import FailureAnalysisProfileNotFoundError

        if profile_id not in self._profile_index:
            raise FailureAnalysisProfileNotFoundError(
                f"Failure benchmark profile not found: {profile_id}"
            )
        return self._profile_index[profile_id]


def compute_benchmark_fingerprint(definition: FailureBenchmarkDefinition) -> str:
    """Produce a deterministic SHA-256 fingerprint of a FailureBenchmarkDefinition."""
    canonical = json.dumps(
        {
            "enabled_metrics": list(definition.enabled_metrics),
            "max_latency_ms": definition.max_latency_ms,
            "determinism_runs": definition.determinism_runs,
            "dataset_version": definition.dataset_version,
        },
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()
