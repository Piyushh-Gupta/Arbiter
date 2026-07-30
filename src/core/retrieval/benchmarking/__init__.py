"""Retrieval Benchmarking & Evaluation subsystem for Arbiter."""

from src.core.retrieval.benchmarking.base import (
    BaseRetrievalBenchmark,
    MetricCalculator,
)
from src.core.retrieval.benchmarking.benchmark_models import (
    AggregateMetrics,
    BenchmarkDataset,
    BenchmarkDefinition,
    BenchmarkEnvironmentMetadata,
    BenchmarkProfile,
    BenchmarkProfileRegistry,
    BenchmarkQuery,
    BenchmarkReport,
    LatencySummary,
    PerQueryMetric,
)
from src.core.retrieval.benchmarking.evaluator import RetrievalEvaluator
from src.core.retrieval.benchmarking.metrics import (
    HitRateCalculator,
    MetricRegistry,
    MRRCalculator,
    NDCGCalculator,
    PrecisionCalculator,
    RecallCalculator,
    compute_hit_rate,
    compute_mrr,
    compute_ndcg,
    compute_precision,
    compute_recall,
)

__all__ = [
    "AggregateMetrics",
    "BaseRetrievalBenchmark",
    "BenchmarkDataset",
    "BenchmarkDefinition",
    "BenchmarkEnvironmentMetadata",
    "BenchmarkProfile",
    "BenchmarkProfileRegistry",
    "BenchmarkQuery",
    "BenchmarkReport",
    "HitRateCalculator",
    "LatencySummary",
    "MRRCalculator",
    "MetricCalculator",
    "MetricRegistry",
    "NDCGCalculator",
    "PerQueryMetric",
    "PrecisionCalculator",
    "RecallCalculator",
    "RetrievalEvaluator",
    "compute_hit_rate",
    "compute_mrr",
    "compute_ndcg",
    "compute_precision",
    "compute_recall",
]
