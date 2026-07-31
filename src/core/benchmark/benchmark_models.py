"""Immutable models for the Benchmarking subsystem."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


class BenchmarkMetricType(str, Enum):
    """Supported benchmarking metric identifiers."""

    ACCURACY = "ACCURACY"
    PRECISION = "PRECISION"
    RECALL = "RECALL"
    F1 = "F1"
    MACRO_F1 = "MACRO_F1"
    MICRO_F1 = "MICRO_F1"
    ECE = "ECE"
    MCE = "MCE"
    BRIER_SCORE = "BRIER_SCORE"
    NEGATIVE_LOG_LIKELIHOOD = "NEGATIVE_LOG_LIKELIHOOD"
    MEAN_LATENCY = "MEAN_LATENCY"
    P95_LATENCY = "P95_LATENCY"
    P99_LATENCY = "P99_LATENCY"
    THROUGHPUT = "THROUGHPUT"
    ABSTENTION_RATE = "ABSTENTION_RATE"
    LOW_CONFIDENCE_RATE = "LOW_CONFIDENCE_RATE"
    CONFLICT_RATE = "CONFLICT_RATE"


class MetricResult(BaseModel):
    """Immutable result of a single metric calculation."""

    metric_type: BenchmarkMetricType = Field(
        ..., description="The type of benchmarking metric."
    )
    value: float = Field(..., description="The calculated float value of the metric.")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Metric metadata/intermediate stats."
    )

    model_config = ConfigDict(frozen=True)


class BenchmarkDefinition(BaseModel):
    """Immutable benchmark execution specification details."""

    benchmark_name: str = Field(
        ..., description="Friendly name of the benchmark suite."
    )
    dataset_identifier: str = Field(
        ..., description="Unique name/identifier of the evaluation dataset."
    )
    selected_metrics: tuple[BenchmarkMetricType, ...] = Field(
        ..., description="Set of metric types to evaluate."
    )
    evaluation_profile_id: str = Field(
        ..., description="Production pipeline profile target ID."
    )

    model_config = ConfigDict(frozen=True)


class BenchmarkResult(BaseModel):
    """Container holding computed metrics, matrices, and metadata."""

    metrics: tuple[MetricResult, ...] = Field(
        ..., description="Calculated evaluation metrics collection."
    )
    confusion_matrix: dict[str, dict[str, int]] = Field(
        ..., description="Confusion matrix dict mapping [actual][predicted] to counts."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Metadata dictionary."
    )

    model_config = ConfigDict(frozen=True)


class BenchmarkTrace(BaseModel):
    """Execution trace logs ensuring exact context reproducibility."""

    dataset_version: str = Field(
        ..., description="Loaded version signature of evaluation dataset."
    )
    execution_sequence: tuple[str, ...] = Field(
        ..., description="Exact evaluation order of record sample IDs."
    )
    metric_execution_order: tuple[BenchmarkMetricType, ...] = Field(
        ..., description="Sequence in which metric calculators were executed."
    )
    configuration_fingerprint: str = Field(
        ..., description="Pipeline configurations hash/fingerprint."
    )
    execution_timestamp: str = Field(
        ..., description="Timezone-aware timestamp string."
    )

    model_config = ConfigDict(frozen=True)


class BenchmarkReport(BaseModel):
    """Result report summarizing outcome, trace details, and system environment info."""

    benchmark_result: BenchmarkResult = Field(
        ..., description="Calculated metrics and matrices."
    )
    benchmark_trace: BenchmarkTrace = Field(
        ..., description="Reproducibility trace metadata."
    )
    execution_environment: dict[str, str] = Field(
        ..., description="Machine CPU, OS, memory, and python details."
    )
    configuration_fingerprint: str = Field(
        ..., description="MD5/SHA256 configuration footprint."
    )

    model_config = ConfigDict(frozen=True)


class BenchmarkProfile(BaseModel):
    """Binding mapping a profile name to its benchmark configuration definition."""

    profile_id: str = Field(..., description="Unique profile registration identifier.")
    definition: BenchmarkDefinition = Field(
        ..., description="The benchmark definition config specifications."
    )

    model_config = ConfigDict(frozen=True)


class BenchmarkProfileRegistry(BaseModel):
    """Registry maintaining active benchmark suite profiles."""

    profiles: tuple[BenchmarkProfile, ...] = Field(
        ..., min_length=1, description="Registered profiles collection."
    )

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def validate_unique_profiles(self) -> "BenchmarkProfileRegistry":
        seen = set()
        for p in self.profiles:
            if p.profile_id in seen:
                raise ValueError(f"Duplicate profile_id '{p.profile_id}' in registry.")
            seen.add(p.profile_id)
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def _by_id(self) -> dict[str, BenchmarkProfile]:
        return {p.profile_id: p for p in self.profiles}

    def resolve(self, profile_id: str) -> BenchmarkProfile:
        if profile_id not in self._by_id:
            raise KeyError(f"Benchmark profile '{profile_id}' not found in registry.")
        return self._by_id[profile_id]
