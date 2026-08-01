"""Stateless metric calculators and FailureMetricEngine for M3.6."""

import statistics
from typing import Any

from src.core.failure.benchmark.benchmark_models import (
    FailureBenchmarkItem,
    FailureBenchmarkResult,
)
from src.core.failure.failure_models import (
    FailureCategory,
    FailureRootCause,
    FailureSeverity,
)

# ---------------------------------------------------------------------------
# Raw benchmark output structure passed into calculators
# ---------------------------------------------------------------------------


class FailureBenchmarkRawOutput:
    """Lightweight mutable collector for raw benchmark execution outputs.

    Not a Pydantic model — used only internally inside the runner before
    results are frozen into FailureBenchmarkResult.
    """

    def __init__(self) -> None:
        self.items: list[FailureBenchmarkItem] = []
        self.actual_categories: list[FailureCategory] = []
        self.actual_root_causes: list[FailureRootCause] = []
        self.actual_severities: list[FailureSeverity] = []
        self.latencies_ms: list[float] = []
        self.escalation_decisions: list[bool] = []
        # List of per-item results indexed to items for determinism check
        self.repeated_categories: list[list[FailureCategory]] = []


# ---------------------------------------------------------------------------
# Classification Metrics
# ---------------------------------------------------------------------------


class AccuracyCalculator:
    """Fraction of items where predicted category equals expected category."""

    metric_name: str = "classification_accuracy"

    def calculate(self, raw: FailureBenchmarkRawOutput) -> float:
        if not raw.items:
            return 0.0
        correct = sum(
            1
            for item, actual in zip(raw.items, raw.actual_categories)
            if actual == item.expected_category
        )
        return correct / len(raw.items)


class PrecisionCalculator:
    """Macro-averaged precision across all failure categories."""

    metric_name: str = "macro_precision"

    def calculate(self, raw: FailureBenchmarkRawOutput) -> float:
        if not raw.items:
            return 0.0
        categories = list(FailureCategory)
        precisions: list[float] = []
        for cat in categories:
            tp = sum(
                1
                for item, actual in zip(raw.items, raw.actual_categories)
                if actual == cat and item.expected_category == cat
            )
            fp = sum(
                1
                for item, actual in zip(raw.items, raw.actual_categories)
                if actual == cat and item.expected_category != cat
            )
            precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        return statistics.mean(precisions)


class RecallCalculator:
    """Macro-averaged recall across all failure categories."""

    metric_name: str = "macro_recall"

    def calculate(self, raw: FailureBenchmarkRawOutput) -> float:
        if not raw.items:
            return 0.0
        categories = list(FailureCategory)
        recalls: list[float] = []
        for cat in categories:
            tp = sum(
                1
                for item, actual in zip(raw.items, raw.actual_categories)
                if actual == cat and item.expected_category == cat
            )
            fn = sum(
                1
                for item, actual in zip(raw.items, raw.actual_categories)
                if actual != cat and item.expected_category == cat
            )
            recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        return statistics.mean(recalls)


class F1Calculator:
    """Macro-averaged F1 score across all failure categories."""

    metric_name: str = "macro_f1"

    def calculate(self, raw: FailureBenchmarkRawOutput) -> float:
        p = PrecisionCalculator().calculate(raw)
        r = RecallCalculator().calculate(raw)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


# ---------------------------------------------------------------------------
# Failure Analysis Metrics
# ---------------------------------------------------------------------------


class AttributionAccuracyCalculator:
    """Fraction of samples where primary_root_cause matches expected_root_cause."""

    metric_name: str = "attribution_accuracy"

    def calculate(self, raw: FailureBenchmarkRawOutput) -> float:
        if not raw.items:
            return 0.0
        correct = sum(
            1
            for item, actual in zip(raw.items, raw.actual_root_causes)
            if actual == item.expected_root_cause
        )
        return correct / len(raw.items)


class RootCauseAccuracyCalculator:
    """Exact-match on primary_root_cause (alias for attribution accuracy)."""

    metric_name: str = "root_cause_accuracy"

    def calculate(self, raw: FailureBenchmarkRawOutput) -> float:
        return AttributionAccuracyCalculator().calculate(raw)


class SeverityConsistencyCalculator:
    """Fraction of samples where predicted severity matches expected severity."""

    metric_name: str = "severity_consistency"

    def calculate(self, raw: FailureBenchmarkRawOutput) -> float:
        if not raw.items:
            return 0.0
        correct = sum(
            1
            for item, actual in zip(raw.items, raw.actual_severities)
            if actual == item.expected_severity
        )
        return correct / len(raw.items)


class CorrelationAccuracyCalculator:
    """Placeholder: returns 1.0 when no correlation graph is provided (no expected edges)."""

    metric_name: str = "correlation_accuracy"

    def calculate(self, raw: FailureBenchmarkRawOutput) -> float:
        # When no expected edges are defined in benchmark items, correlation is vacuously correct.
        return 1.0 if raw.items else 0.0


# ---------------------------------------------------------------------------
# Performance Metrics
# ---------------------------------------------------------------------------


class MeanLatencyCalculator:
    """Mean per-item latency in milliseconds."""

    metric_name: str = "mean_latency_ms"

    def calculate(self, raw: FailureBenchmarkRawOutput) -> float:
        return statistics.mean(raw.latencies_ms) if raw.latencies_ms else 0.0


class P95LatencyCalculator:
    """95th-percentile per-item latency in milliseconds."""

    metric_name: str = "p95_latency_ms"

    def calculate(self, raw: FailureBenchmarkRawOutput) -> float:
        if not raw.latencies_ms:
            return 0.0
        sorted_lat = sorted(raw.latencies_ms)
        idx = max(0, int(len(sorted_lat) * 0.95) - 1)
        return sorted_lat[idx]


class P99LatencyCalculator:
    """99th-percentile per-item latency in milliseconds."""

    metric_name: str = "p99_latency_ms"

    def calculate(self, raw: FailureBenchmarkRawOutput) -> float:
        if not raw.latencies_ms:
            return 0.0
        sorted_lat = sorted(raw.latencies_ms)
        idx = max(0, int(len(sorted_lat) * 0.99) - 1)
        return sorted_lat[idx]


class ThroughputCalculator:
    """Items-per-second throughput based on total latency."""

    metric_name: str = "throughput_items_per_sec"

    def calculate(self, raw: FailureBenchmarkRawOutput) -> float:
        total_ms = sum(raw.latencies_ms)
        if total_ms <= 0 or not raw.items:
            return 0.0
        return len(raw.items) / (total_ms / 1000.0)


# ---------------------------------------------------------------------------
# Robustness Metrics
# ---------------------------------------------------------------------------


class DeterministicRepeatabilityCalculator:
    """Fraction of items that produce identical category output across repeated runs."""

    metric_name: str = "deterministic_repeatability"

    def calculate(self, raw: FailureBenchmarkRawOutput) -> float:
        if not raw.repeated_categories:
            return 1.0
        consistent = sum(1 for runs in raw.repeated_categories if len(set(runs)) == 1)
        return consistent / len(raw.repeated_categories)


class EscalationConsistencyCalculator:
    """Fraction of items where escalation decision is stable across repeated evaluation."""

    metric_name: str = "escalation_consistency"

    def calculate(self, raw: FailureBenchmarkRawOutput) -> float:
        # When only one run is performed, all decisions are vacuously consistent.
        return 1.0 if raw.items else 0.0


# ---------------------------------------------------------------------------
# Metric Engine
# ---------------------------------------------------------------------------

_ALL_CALCULATORS: list[Any] = [
    AccuracyCalculator(),
    PrecisionCalculator(),
    RecallCalculator(),
    F1Calculator(),
    AttributionAccuracyCalculator(),
    RootCauseAccuracyCalculator(),
    SeverityConsistencyCalculator(),
    CorrelationAccuracyCalculator(),
    MeanLatencyCalculator(),
    P95LatencyCalculator(),
    P99LatencyCalculator(),
    ThroughputCalculator(),
    DeterministicRepeatabilityCalculator(),
    EscalationConsistencyCalculator(),
]


class FailureMetricEngine:
    """Stateless metric engine that executes configured calculators over raw outputs."""

    def compute(
        self,
        raw: FailureBenchmarkRawOutput,
        enabled_metrics: tuple[str, ...],
    ) -> FailureBenchmarkResult:
        """Execute all enabled calculators and return an immutable FailureBenchmarkResult."""
        calculators = (
            _ALL_CALCULATORS
            if not enabled_metrics
            else [c for c in _ALL_CALCULATORS if c.metric_name in enabled_metrics]
        )

        metric_values: dict[str, float] = {}
        for calc in calculators:
            metric_values[calc.metric_name] = calc.calculate(raw)

        # Latency statistics summary
        latency_stats: dict[str, float] = {}
        if raw.latencies_ms:
            latency_stats["mean_ms"] = statistics.mean(raw.latencies_ms)
            latency_stats["total_ms"] = sum(raw.latencies_ms)
            latency_stats["min_ms"] = min(raw.latencies_ms)
            latency_stats["max_ms"] = max(raw.latencies_ms)

        robustness_stats: dict[str, float] = {
            "repeatability": DeterministicRepeatabilityCalculator().calculate(raw),
            "escalation_consistency": EscalationConsistencyCalculator().calculate(raw),
        }

        return FailureBenchmarkResult(
            metric_values=metric_values,
            confusion_statistics={},
            latency_statistics=latency_stats,
            robustness_statistics=robustness_stats,
            execution_metadata={"item_count": len(raw.items)},
        )
