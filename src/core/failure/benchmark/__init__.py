"""Failure Analysis Benchmarking subsystem (M3.6)."""

from src.core.failure.benchmark.base import (
    BaseFailureBenchmarkDataset,
    BaseFailureBenchmarkRunner,
    BaseMetricCalculator,
)
from src.core.failure.benchmark.benchmark_models import (
    FailureBenchmarkDefinition,
    FailureBenchmarkItem,
    FailureBenchmarkProfile,
    FailureBenchmarkProfileRegistry,
    FailureBenchmarkReport,
    FailureBenchmarkResult,
    FailureBenchmarkSuite,
    compute_benchmark_fingerprint,
)
from src.core.failure.benchmark.metrics import (
    AccuracyCalculator,
    AttributionAccuracyCalculator,
    CorrelationAccuracyCalculator,
    DeterministicRepeatabilityCalculator,
    EscalationConsistencyCalculator,
    F1Calculator,
    FailureBenchmarkRawOutput,
    FailureMetricEngine,
    MeanLatencyCalculator,
    P95LatencyCalculator,
    P99LatencyCalculator,
    PrecisionCalculator,
    RecallCalculator,
    RootCauseAccuracyCalculator,
    SeverityConsistencyCalculator,
    ThroughputCalculator,
)
from src.core.failure.benchmark.runner import FailureBenchmarkRunner

__all__ = [
    # Protocols
    "BaseFailureBenchmarkDataset",
    "BaseMetricCalculator",
    "BaseFailureBenchmarkRunner",
    # Models
    "FailureBenchmarkItem",
    "FailureBenchmarkDefinition",
    "FailureBenchmarkSuite",
    "FailureBenchmarkResult",
    "FailureBenchmarkReport",
    "FailureBenchmarkProfile",
    "FailureBenchmarkProfileRegistry",
    "compute_benchmark_fingerprint",
    # Metrics
    "FailureBenchmarkRawOutput",
    "FailureMetricEngine",
    "AccuracyCalculator",
    "PrecisionCalculator",
    "RecallCalculator",
    "F1Calculator",
    "AttributionAccuracyCalculator",
    "RootCauseAccuracyCalculator",
    "SeverityConsistencyCalculator",
    "CorrelationAccuracyCalculator",
    "MeanLatencyCalculator",
    "P95LatencyCalculator",
    "P99LatencyCalculator",
    "ThroughputCalculator",
    "DeterministicRepeatabilityCalculator",
    "EscalationConsistencyCalculator",
    # Runner
    "FailureBenchmarkRunner",
]
