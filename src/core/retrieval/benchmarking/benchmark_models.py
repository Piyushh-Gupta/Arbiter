"""Immutable domain models for the Retrieval Benchmarking & Evaluation subsystem."""

import hashlib
import json
import typing
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

if typing.TYPE_CHECKING:
    from src.core.retrieval.benchmarking.base import BaseRetrievalBenchmark
else:
    BaseRetrievalBenchmark = typing.Any

__all__ = [
    "AggregateMetrics",
    "BenchmarkDataset",
    "BenchmarkDefinition",
    "BenchmarkEnvironmentMetadata",
    "BenchmarkProfile",
    "BenchmarkProfileRegistry",
    "BenchmarkQuery",
    "BenchmarkReport",
    "LatencySummary",
    "PerQueryMetric",
]


class BenchmarkQuery(BaseModel):
    """Immutable single query with ground truth evidence."""

    query_id: str = Field(..., description="Unique query identifier.")
    query_text: str = Field(..., description="Query or claim text.")
    expected_span_ids: tuple[str, ...] = Field(
        ..., description="Tuple of ground truth span IDs expected to be retrieved."
    )
    expected_document_ids: tuple[str, ...] = Field(
        default=(), description="Optional tuple of ground truth document IDs."
    )
    metadata: Mapping[str, Any] = Field(
        default_factory=dict, description="Extensible query metadata."
    )

    model_config = ConfigDict(frozen=True)


class BenchmarkDataset(BaseModel):
    """Immutable collection of benchmark queries with an embedded dataset fingerprint."""

    benchmark_id: str = Field(..., description="Unique benchmark dataset identifier.")
    dataset_version: str = Field(..., description="Version of the benchmark dataset.")
    dataset_fingerprint: str = Field(
        default="",
        description="SHA-256 fingerprint uniquely identifying dataset content.",
    )
    queries: tuple[BenchmarkQuery, ...] = Field(
        ..., min_length=1, description="Ordered tuple of benchmark queries."
    )
    metadata: Mapping[str, Any] = Field(
        default_factory=dict, description="Dataset-level metadata."
    )

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def _ensure_fingerprint(self) -> "BenchmarkDataset":
        if not self.dataset_fingerprint:
            canonical_items = [
                {
                    "query_id": q.query_id,
                    "query_text": q.query_text,
                    "expected_span_ids": list(q.expected_span_ids),
                    "expected_document_ids": list(q.expected_document_ids),
                }
                for q in self.queries
            ]
            raw = json.dumps(
                {
                    "benchmark_id": self.benchmark_id,
                    "dataset_version": self.dataset_version,
                    "queries": canonical_items,
                },
                sort_keys=True,
            )
            fp = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            object.__setattr__(self, "dataset_fingerprint", fp)
        return self


class BenchmarkEnvironmentMetadata(BaseModel):
    """Immutable metadata describing the benchmark execution environment."""

    retrieval_profile_id: str = Field(..., description="Retrieval profile identifier.")
    reranking_profile_id: str | None = Field(
        default=None, description="Optional reranking profile identifier."
    )
    cache_profile_id: str | None = Field(
        default=None, description="Optional cache profile identifier."
    )
    corpus_version: str = Field(..., description="Corpus/index version identifier.")
    benchmark_dataset_fingerprint: str = Field(
        ..., description="Dataset SHA-256 fingerprint."
    )
    retrieval_model_metadata: Mapping[str, str] = Field(
        default_factory=dict, description="Retrieval model parameters."
    )
    reranking_model_metadata: Mapping[str, str] = Field(
        default_factory=dict, description="Reranking model parameters."
    )
    execution_device: str = Field(
        default="cpu", description="Execution device (e.g. 'cpu', 'cuda')."
    )

    model_config = ConfigDict(frozen=True)


class BenchmarkDefinition(BaseModel):
    """Immutable configuration for a benchmark execution run."""

    top_k: int = Field(
        default=5, gt=0, description="Top K cutoff parameter for evaluation."
    )
    metrics: tuple[str, ...] = Field(
        default=("recall", "precision", "mrr", "ndcg", "hit_rate"),
        description="Metric identifiers to evaluate.",
    )
    compute_latency_percentiles: bool = Field(
        default=True, description="Whether to compute detailed latency summary."
    )

    model_config = ConfigDict(frozen=True)


class PerQueryMetric(BaseModel):
    """Immutable evaluation metrics for a single query."""

    query_id: str = Field(..., description="Query identifier.")
    recall: float = Field(..., ge=0.0, le=1.0)
    precision: float = Field(..., ge=0.0, le=1.0)
    mrr: float = Field(..., ge=0.0, le=1.0)
    ndcg: float = Field(..., ge=0.0, le=1.0)
    hit: bool = Field(...)
    latency_ms: float = Field(..., ge=0.0)
    cache_hit: bool = Field(default=False)
    metrics: Mapping[str, float] = Field(
        default_factory=dict, description="Extensible metrics dict."
    )

    model_config = ConfigDict(frozen=True)


class AggregateMetrics(BaseModel):
    """Immutable summary statistics across all benchmark queries."""

    mean_recall: float = Field(...)
    mean_precision: float = Field(...)
    mrr: float = Field(...)
    mean_ndcg: float = Field(...)
    hit_rate: float = Field(...)
    metrics: Mapping[str, float] = Field(
        default_factory=dict, description="Extensible metrics dict."
    )

    model_config = ConfigDict(frozen=True)


class LatencySummary(BaseModel):
    """Immutable latency metrics summary computed via monotonic timing."""

    min_ms: float = Field(..., ge=0.0)
    max_ms: float = Field(..., ge=0.0)
    p50_ms: float = Field(..., ge=0.0)
    p90_ms: float = Field(..., ge=0.0)
    p99_ms: float = Field(..., ge=0.0)
    mean_ms: float = Field(..., ge=0.0)

    model_config = ConfigDict(frozen=True)


class BenchmarkReport(BaseModel):
    """Immutable comprehensive evaluation report."""

    benchmark_id: str = Field(...)
    timestamp: float = Field(...)
    corpus_version: str = Field(...)
    dataset_fingerprint: str = Field(...)
    environment_metadata: BenchmarkEnvironmentMetadata = Field(...)
    total_queries: int = Field(..., gt=0)
    aggregate_metrics: AggregateMetrics = Field(...)
    latency_summary: LatencySummary = Field(...)
    cache_hit_ratio: float = Field(..., ge=0.0, le=1.0)
    per_query_metrics: tuple[PerQueryMetric, ...] = Field(...)

    model_config = ConfigDict(frozen=True)


class BenchmarkProfile(BaseModel):
    """Immutable reusable wrapper binding a benchmark definition to its execution strategy."""

    profile_id: str = Field(
        ..., description="Unique identifier for this benchmark profile."
    )
    definition: BenchmarkDefinition = Field(
        ..., description="Immutable configuration for benchmark strategy."
    )
    strategy: BaseRetrievalBenchmark = Field(
        ..., description="The executable benchmark evaluation strategy."
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class BenchmarkProfileRegistry(BaseModel):
    """Immutable namespace for securely resolving named benchmark profiles in O(1) time."""

    profiles: tuple[BenchmarkProfile, ...] = Field(
        ...,
        min_length=1,
        description="The collection of registered benchmark profiles.",
    )

    _profile_index: dict[str, BenchmarkProfile] = PrivateAttr(default_factory=dict)

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _build_and_validate_index(self) -> "BenchmarkProfileRegistry":
        from src.core.exceptions import DuplicateBenchmarkProfileError

        index: dict[str, BenchmarkProfile] = {}
        for profile in self.profiles:
            if profile.profile_id in index:
                raise DuplicateBenchmarkProfileError(
                    f"Duplicate benchmark profile identifier: {profile.profile_id}"
                )
            index[profile.profile_id] = profile

        object.__setattr__(self, "_profile_index", index)
        return self

    def resolve(self, profile_id: str) -> BenchmarkProfile:
        from src.core.exceptions import BenchmarkProfileNotFoundError

        if profile_id not in self._profile_index:
            raise BenchmarkProfileNotFoundError(
                f"Benchmark profile not found: {profile_id}"
            )
        return self._profile_index[profile_id]
