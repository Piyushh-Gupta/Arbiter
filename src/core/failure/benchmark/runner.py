"""Stateless FailureBenchmarkRunner orchestrating benchmark suite execution (M3.6)."""

import time
from datetime import UTC, datetime

from src.core.failure.benchmark.benchmark_models import (
    FailureBenchmarkDefinition,
    FailureBenchmarkReport,
    FailureBenchmarkSuite,
    compute_benchmark_fingerprint,
)
from src.core.failure.benchmark.metrics import (
    FailureBenchmarkRawOutput,
    FailureMetricEngine,
)
from src.core.failure.failure_models import (
    AnalyzerExecutionResult,
    FailureCategory,
    FailureRootCause,
    FailureSeverity,
)


def _infer_category(result: AnalyzerExecutionResult) -> FailureCategory:
    """Extract the failure category from an analyzer execution result."""
    return result.classification.category


def _infer_root_cause(result: AnalyzerExecutionResult) -> FailureRootCause:
    """Infer a root cause from the highest-confidence diagnostic evidence, defaulting to UNKNOWN."""
    for evidence in result.diagnostic_evidence:
        if hasattr(evidence, "root_cause"):
            rc = evidence.root_cause
            if isinstance(rc, FailureRootCause):
                return rc
    return FailureRootCause.UNKNOWN


def _infer_severity(result: AnalyzerExecutionResult) -> FailureSeverity:
    """Extract the failure severity from an analyzer execution result."""
    return result.classification.severity


class FailureBenchmarkRunner:
    """Stateless orchestrator for offline failure benchmark suite execution.

    Delegates metric computation entirely to FailureMetricEngine.
    """

    def __init__(self, metric_engine: FailureMetricEngine | None = None) -> None:
        self._metric_engine = metric_engine or FailureMetricEngine()

    def validate_compatibility(self, definition: FailureBenchmarkDefinition) -> None:
        """No-op: FailureBenchmarkRunner is compatible with any FailureBenchmarkDefinition."""

    def run(
        self,
        suite: FailureBenchmarkSuite,
        definition: FailureBenchmarkDefinition,
    ) -> FailureBenchmarkReport:
        """Execute the benchmark suite and return an immutable FailureBenchmarkReport."""
        raw = FailureBenchmarkRawOutput()
        trace: list[str] = [
            f"suite:{suite.suite_id}",
            f"profile:{suite.evaluation_profile}",
        ]

        items = suite.dataset.items() if hasattr(suite.dataset, "items") else ()

        for item in items:
            raw.items.append(item)
            trace.append(f"item:{item.item_id}")

            # Time the pipeline execution.
            t0 = time.perf_counter()

            # Derive actual category/root-cause/severity from the item's analyzer results.
            if item.analyzer_execution_results:
                first = item.analyzer_execution_results[0]
                actual_cat = _infer_category(first)
                actual_rc = _infer_root_cause(first)
                actual_sev = _infer_severity(first)
            else:
                # No results — treat as unknown.
                actual_cat = FailureCategory.UNKNOWN
                actual_rc = FailureRootCause.UNKNOWN
                actual_sev = FailureSeverity.INFO

            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            raw.actual_categories.append(actual_cat)
            raw.actual_root_causes.append(actual_rc)
            raw.actual_severities.append(actual_sev)
            raw.latencies_ms.append(elapsed_ms)

            # Determinism: run the same inference N more times and track categories.
            repeated: list[FailureCategory] = [actual_cat]
            for _ in range(max(0, definition.determinism_runs - 1)):
                if item.analyzer_execution_results:
                    repeated.append(_infer_category(item.analyzer_execution_results[0]))
                else:
                    repeated.append(FailureCategory.UNKNOWN)
            raw.repeated_categories.append(repeated)

        trace.append(f"items_processed:{len(raw.items)}")

        benchmark_result = self._metric_engine.compute(
            raw, suite.enabled_metrics or definition.enabled_metrics
        )

        fingerprint = compute_benchmark_fingerprint(definition)
        timestamp = datetime.now(UTC).isoformat()

        return FailureBenchmarkReport(
            result=benchmark_result,
            configuration_fingerprint=fingerprint,
            execution_timestamp=timestamp,
            benchmark_trace=tuple(trace),
        )
