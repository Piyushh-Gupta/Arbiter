"""Concrete decision metric calculators and metric engine implementations (M4.5)."""

import statistics
from typing import Sequence

from src.core.decision.benchmark.base import BaseDecisionMetricCalculator
from src.core.decision.benchmark.benchmark_models import (
    DecisionBenchmarkMetrics,
    DecisionBenchmarkRawOutput,
)


class DecisionAccuracyCalculator(BaseDecisionMetricCalculator):
    """Computes ratio of actual decision actions matching the expected ground truth."""

    @property
    def metric_name(self) -> str:
        return "accuracy"

    def calculate(self, raw_output: DecisionBenchmarkRawOutput) -> float:
        if not raw_output.expected_actions:
            return 0.0

        matches = sum(
            1
            for exp, act in zip(raw_output.expected_actions, raw_output.actual_actions)
            if exp == act
        )
        return float(matches / len(raw_output.expected_actions))


class AbstentionRateCalculator(BaseDecisionMetricCalculator):
    """Computes ratio of items resulting in the ABSTAIN action."""

    @property
    def metric_name(self) -> str:
        return "abstention_rate"

    def calculate(self, raw_output: DecisionBenchmarkRawOutput) -> float:
        if not raw_output.actual_actions:
            return 0.0

        abstentions = sum(1 for act in raw_output.actual_actions if act == "ABSTAIN")
        return float(abstentions / len(raw_output.actual_actions))


class EscalationRateCalculator(BaseDecisionMetricCalculator):
    """Computes ratio of items resulting in the ESCALATE action."""

    @property
    def metric_name(self) -> str:
        return "escalation_rate"

    def calculate(self, raw_output: DecisionBenchmarkRawOutput) -> float:
        if not raw_output.actual_actions:
            return 0.0

        escalations = sum(1 for act in raw_output.actual_actions if act == "ESCALATE")
        return float(escalations / len(raw_output.actual_actions))


class MeanLatencyCalculator(BaseDecisionMetricCalculator):
    """Computes average execution latency in milliseconds."""

    @property
    def metric_name(self) -> str:
        return "mean_latency_ms"

    def calculate(self, raw_output: DecisionBenchmarkRawOutput) -> float:
        if not raw_output.latencies_ms:
            return 0.0
        return float(statistics.mean(raw_output.latencies_ms))


class ThroughputCalculator(BaseDecisionMetricCalculator):
    """Computes throughput in queries per second (QPS) over raw items."""

    @property
    def metric_name(self) -> str:
        return "throughput_qps"

    def calculate(self, raw_output: DecisionBenchmarkRawOutput) -> float:
        if not raw_output.latencies_ms:
            return 0.0

        total_time_seconds = sum(raw_output.latencies_ms) / 1000.0
        if total_time_seconds <= 0.0:
            return 0.0

        return float(len(raw_output.latencies_ms) / total_time_seconds)


_ALL_CALCULATORS: Sequence[BaseDecisionMetricCalculator] = (
    DecisionAccuracyCalculator(),
    AbstentionRateCalculator(),
    EscalationRateCalculator(),
    MeanLatencyCalculator(),
    ThroughputCalculator(),
)


class DecisionMetricEngine:
    """Stateless metric engine executing enabled calculators over raw outputs to construct DecisionBenchmarkMetrics."""

    def compute(
        self,
        raw_output: DecisionBenchmarkRawOutput,
        enabled_metrics: tuple[str, ...] | None = None,
    ) -> DecisionBenchmarkMetrics:
        """Executes all enabled calculators and produces a strongly typed DecisionBenchmarkMetrics."""
        # Find which calculators to run
        active_names = (
            set(enabled_metrics)
            if enabled_metrics
            else {
                "accuracy",
                "abstention_rate",
                "escalation_rate",
                "mean_latency_ms",
                "throughput_qps",
            }
        )

        results: dict[str, float] = {}
        for calc in _ALL_CALCULATORS:
            if calc.metric_name in active_names:
                results[calc.metric_name] = calc.calculate(raw_output)
            else:
                # Default missing metric values to 0.0
                results[calc.metric_name] = 0.0

        return DecisionBenchmarkMetrics(
            accuracy=results.get("accuracy", 0.0),
            abstention_rate=results.get("abstention_rate", 0.0),
            escalation_rate=results.get("escalation_rate", 0.0),
            mean_latency_ms=results.get("mean_latency_ms", 0.0),
            throughput_qps=results.get("throughput_qps", 0.0),
        )
