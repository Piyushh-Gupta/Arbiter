"""Metric calculators and metric engine for Pipeline Benchmarking & Evaluation (M5.4)."""

import math
import statistics
from typing import Sequence

from src.core.pipeline.benchmark.base import BasePipelineMetricCalculator
from src.core.pipeline.benchmark.benchmark_models import (
    PipelineBenchmarkMetric,
    PipelineBenchmarkMetrics,
    PipelineBenchmarkRawOutput,
    PipelineStageBenchmarkMetrics,
)


def _percentile(latencies: Sequence[float], p: float) -> float:
    """Computes percentile index-based latency value deterministically."""
    if not latencies:
        return 0.0
    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)
    idx = math.floor((p / 100.0) * n)
    idx = max(0, min(idx, n - 1))
    return float(sorted_latencies[idx])


class PipelineSuccessRateCalculator(BasePipelineMetricCalculator):
    """Computes success rate: sum(actual_successes) / total_items."""

    @property
    def metric_name(self) -> str:
        return PipelineBenchmarkMetric.SUCCESS_RATE.value

    def calculate(self, raw_output: PipelineBenchmarkRawOutput) -> float:
        if not raw_output.actual_successes:
            return 0.0
        successes = sum(1 for s in raw_output.actual_successes if s)
        return float(successes / len(raw_output.actual_successes))


class PipelineMeanLatencyCalculator(BasePipelineMetricCalculator):
    """Computes mean end-to-end pipeline latency in milliseconds."""

    @property
    def metric_name(self) -> str:
        return PipelineBenchmarkMetric.MEAN_LATENCY_MS.value

    def calculate(self, raw_output: PipelineBenchmarkRawOutput) -> float:
        if not raw_output.total_latencies_ms:
            return 0.0
        return float(statistics.mean(raw_output.total_latencies_ms))


class PipelineP50LatencyCalculator(BasePipelineMetricCalculator):
    """Computes 50th percentile latency (median) deterministically."""

    @property
    def metric_name(self) -> str:
        return PipelineBenchmarkMetric.P50_LATENCY_MS.value

    def calculate(self, raw_output: PipelineBenchmarkRawOutput) -> float:
        return _percentile(raw_output.total_latencies_ms, 50.0)


class PipelineP95LatencyCalculator(BasePipelineMetricCalculator):
    """Computes 95th percentile latency deterministically."""

    @property
    def metric_name(self) -> str:
        return PipelineBenchmarkMetric.P95_LATENCY_MS.value

    def calculate(self, raw_output: PipelineBenchmarkRawOutput) -> float:
        return _percentile(raw_output.total_latencies_ms, 95.0)


class PipelineP99LatencyCalculator(BasePipelineMetricCalculator):
    """Computes 99th percentile latency deterministically."""

    @property
    def metric_name(self) -> str:
        return PipelineBenchmarkMetric.P99_LATENCY_MS.value

    def calculate(self, raw_output: PipelineBenchmarkRawOutput) -> float:
        return _percentile(raw_output.total_latencies_ms, 99.0)


class PipelineThroughputCalculator(BasePipelineMetricCalculator):
    """Computes throughput in queries per second (QPS)."""

    @property
    def metric_name(self) -> str:
        return PipelineBenchmarkMetric.THROUGHPUT_QPS.value

    def calculate(self, raw_output: PipelineBenchmarkRawOutput) -> float:
        if not raw_output.total_latencies_ms:
            return 0.0
        total_time_seconds = sum(raw_output.total_latencies_ms) / 1000.0
        if total_time_seconds <= 0.0:
            return 0.0
        return float(len(raw_output.total_latencies_ms) / total_time_seconds)


class PipelineRetryRateCalculator(BasePipelineMetricCalculator):
    """Computes ratio of items that required retries (attempts > 1)."""

    @property
    def metric_name(self) -> str:
        return PipelineBenchmarkMetric.RETRY_RATE.value

    def calculate(self, raw_output: PipelineBenchmarkRawOutput) -> float:
        if not raw_output.retry_attempt_counts:
            return 0.0
        retried = sum(1 for attempts in raw_output.retry_attempt_counts if attempts > 1)
        return float(retried / len(raw_output.retry_attempt_counts))


class PipelineMeanRetryAttemptsCalculator(BasePipelineMetricCalculator):
    """Computes the mean number of attempts across all items."""

    @property
    def metric_name(self) -> str:
        return PipelineBenchmarkMetric.MEAN_RETRY_ATTEMPTS.value

    def calculate(self, raw_output: PipelineBenchmarkRawOutput) -> float:
        if not raw_output.retry_attempt_counts:
            return 0.0
        return float(statistics.mean(raw_output.retry_attempt_counts))


class PipelineTimeoutRateCalculator(BasePipelineMetricCalculator):
    """Computes ratio of items where a timeout was triggered."""

    @property
    def metric_name(self) -> str:
        return PipelineBenchmarkMetric.TIMEOUT_RATE.value

    def calculate(self, raw_output: PipelineBenchmarkRawOutput) -> float:
        if not raw_output.timeout_triggered:
            return 0.0
        timeouts = sum(1 for t in raw_output.timeout_triggered if t)
        return float(timeouts / len(raw_output.timeout_triggered))


class PipelineRecoveryRateCalculator(BasePipelineMetricCalculator):
    """Computes ratio of items where recovery strategy was invoked."""

    @property
    def metric_name(self) -> str:
        return PipelineBenchmarkMetric.RECOVERY_RATE.value

    def calculate(self, raw_output: PipelineBenchmarkRawOutput) -> float:
        if not raw_output.recovery_invoked:
            return 0.0
        recoveries = sum(1 for r in raw_output.recovery_invoked if r)
        return float(recoveries / len(raw_output.recovery_invoked))


class PipelineDeterminismRateCalculator(BasePipelineMetricCalculator):
    """Computes determinism rate over repeated claims in the dataset."""

    @property
    def metric_name(self) -> str:
        return PipelineBenchmarkMetric.DETERMINISM_RATE.value

    def calculate(self, raw_output: PipelineBenchmarkRawOutput) -> float:
        if not raw_output.claims or not raw_output.actual_successes:
            return 1.0

        claim_to_outcomes: dict[str, list[bool]] = {}
        for claim, success in zip(raw_output.claims, raw_output.actual_successes):
            claim_to_outcomes.setdefault(claim, []).append(success)

        multi_occurrence_claims = {
            c: outcomes
            for c, outcomes in claim_to_outcomes.items()
            if len(outcomes) > 1
        }
        if not multi_occurrence_claims:
            return 1.0

        deterministic_claims_count = sum(
            1
            for outcomes in multi_occurrence_claims.values()
            if len(set(outcomes)) == 1
        )
        return float(deterministic_claims_count / len(multi_occurrence_claims))


_ALL_CALCULATORS: Sequence[BasePipelineMetricCalculator] = (
    PipelineSuccessRateCalculator(),
    PipelineMeanLatencyCalculator(),
    PipelineP50LatencyCalculator(),
    PipelineP95LatencyCalculator(),
    PipelineP99LatencyCalculator(),
    PipelineThroughputCalculator(),
    PipelineRetryRateCalculator(),
    PipelineMeanRetryAttemptsCalculator(),
    PipelineTimeoutRateCalculator(),
    PipelineRecoveryRateCalculator(),
    PipelineDeterminismRateCalculator(),
)


class PipelineBenchmarkMetricEngine:
    """Stateless metric engine computing all enabled metrics from raw outputs."""

    def compute(
        self,
        raw_output: PipelineBenchmarkRawOutput,
        enabled_metrics: tuple[PipelineBenchmarkMetric, ...] | None = None,
    ) -> PipelineBenchmarkMetrics:
        """Executes all enabled calculators, default missing metric values to 0.0."""
        active_names = (
            {m.value for m in enabled_metrics}
            if enabled_metrics
            else {m.value for m in PipelineBenchmarkMetric}
        )

        results = {}
        for calc in _ALL_CALCULATORS:
            if calc.metric_name in active_names:
                results[calc.metric_name] = calc.calculate(raw_output)
            else:
                results[calc.metric_name] = 0.0

        return PipelineBenchmarkMetrics(
            success_rate=results.get(PipelineBenchmarkMetric.SUCCESS_RATE.value, 0.0),
            mean_latency_ms=results.get(
                PipelineBenchmarkMetric.MEAN_LATENCY_MS.value, 0.0
            ),
            p50_latency_ms=results.get(
                PipelineBenchmarkMetric.P50_LATENCY_MS.value, 0.0
            ),
            p95_latency_ms=results.get(
                PipelineBenchmarkMetric.P95_LATENCY_MS.value, 0.0
            ),
            p99_latency_ms=results.get(
                PipelineBenchmarkMetric.P99_LATENCY_MS.value, 0.0
            ),
            throughput_qps=results.get(
                PipelineBenchmarkMetric.THROUGHPUT_QPS.value, 0.0
            ),
            retry_rate=results.get(PipelineBenchmarkMetric.RETRY_RATE.value, 0.0),
            mean_retry_attempts=results.get(
                PipelineBenchmarkMetric.MEAN_RETRY_ATTEMPTS.value, 0.0
            ),
            timeout_rate=results.get(PipelineBenchmarkMetric.TIMEOUT_RATE.value, 0.0),
            recovery_rate=results.get(PipelineBenchmarkMetric.RECOVERY_RATE.value, 0.0),
            determinism_rate=results.get(
                PipelineBenchmarkMetric.DETERMINISM_RATE.value, 0.0
            ),
        )

    def compute_stage_metrics(
        self,
        raw_output: PipelineBenchmarkRawOutput,
    ) -> dict[str, PipelineStageBenchmarkMetrics]:
        """Aggregates per-stage latencies and returns stage benchmark metrics with O(1) lookup."""
        stage_to_latencies: dict[str, list[float]] = {}
        for stage_dict in raw_output.stage_latencies_ms:
            for stage_id, latency in stage_dict.items():
                stage_to_latencies.setdefault(stage_id, []).append(latency)

        stage_metrics = {}
        for stage_id, latencies in stage_to_latencies.items():
            if not latencies:
                continue
            mean_latency = float(statistics.mean(latencies))
            p50 = _percentile(latencies, 50.0)
            p95 = _percentile(latencies, 95.0)
            p99 = _percentile(latencies, 99.0)
            stage_metrics[stage_id] = PipelineStageBenchmarkMetrics(
                stage_id=stage_id,
                mean_latency_ms=mean_latency,
                p50_latency_ms=p50,
                p95_latency_ms=p95,
                p99_latency_ms=p99,
            )
        return stage_metrics
