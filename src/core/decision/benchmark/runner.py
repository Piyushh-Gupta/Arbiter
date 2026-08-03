"""Executor and Runner implementations for Decision Benchmarking & Evaluation Framework (M4.5)."""

import statistics
import time
from datetime import UTC, datetime
from typing import Any

from src.core.decision.benchmark.base import (
    BaseDecisionBenchmarkExecutor,
    BaseDecisionBenchmarkRunner,
)
from src.core.decision.benchmark.benchmark_models import (
    DecisionBenchmarkDataset,
    DecisionBenchmarkProfile,
    DecisionBenchmarkRawOutput,
    DecisionBenchmarkReport,
    DecisionBenchmarkResult,
    DecisionBenchmarkSuite,
)
from src.core.decision.benchmark.metrics import DecisionMetricEngine


class DecisionBenchmarkExecutor(BaseDecisionBenchmarkExecutor):
    """Stateless executor executing decision strategy over a dataset to generate raw execution traces."""

    def execute(
        self, dataset: DecisionBenchmarkDataset, strategy: Any
    ) -> DecisionBenchmarkRawOutput:
        """Executes the decision strategy over all items, measuring individual latency."""
        item_ids: list[str] = []
        expected_actions: list[str] = []
        actual_actions: list[str] = []
        latencies_ms: list[float] = []
        decisions: list[Any] = []

        for item in dataset.items:
            start_time = time.perf_counter()
            decision_res = strategy.decide(item.context)
            end_time = time.perf_counter()

            latency = (end_time - start_time) * 1000.0

            item_ids.append(item.item_id)
            expected_actions.append(item.expected_action)
            # Normalize final verdict string
            actual_action = (
                decision_res.final_verdict.value
                if hasattr(decision_res.final_verdict, "value")
                else str(decision_res.final_verdict)
            )
            actual_actions.append(actual_action)
            latencies_ms.append(latency)
            decisions.append(decision_res)

        return DecisionBenchmarkRawOutput(
            suite_id=dataset.dataset_id,
            item_ids=tuple(item_ids),
            expected_actions=tuple(expected_actions),
            actual_actions=tuple(actual_actions),
            latencies_ms=tuple(latencies_ms),
            decisions=tuple(decisions),
        )


class DecisionBenchmarkRunner(BaseDecisionBenchmarkRunner):
    """Stateless orchestrator that coordinates benchmark execution, metric evaluation, and report building."""

    def __init__(
        self,
        strategy: Any,
        executor: BaseDecisionBenchmarkExecutor | None = None,
        metric_engine: DecisionMetricEngine | None = None,
    ) -> None:
        self.strategy = strategy
        self.executor = executor or DecisionBenchmarkExecutor()
        self.metric_engine = metric_engine or DecisionMetricEngine()

    def run(
        self, suite: DecisionBenchmarkSuite, profile: DecisionBenchmarkProfile
    ) -> DecisionBenchmarkReport:
        """Runs the offline benchmark suite and produces an immutable DecisionBenchmarkReport."""
        # 1. Execute raw run
        raw_output = self.executor.execute(suite.dataset, self.strategy)

        # 2. Compute metrics
        metrics = self.metric_engine.compute(
            raw_output, enabled_metrics=profile.enabled_metrics
        )

        # 3. Calculate latency stats
        latencies = raw_output.latencies_ms
        latency_stats: dict[str, float] = {}
        if latencies:
            latency_stats["min_ms"] = float(min(latencies))
            latency_stats["max_ms"] = float(max(latencies))
            latency_stats["mean_ms"] = float(statistics.mean(latencies))
            if len(latencies) > 1:
                latency_stats["stddev_ms"] = float(statistics.stdev(latencies))
            else:
                latency_stats["stddev_ms"] = 0.0
        else:
            latency_stats["min_ms"] = 0.0
            latency_stats["max_ms"] = 0.0
            latency_stats["mean_ms"] = 0.0
            latency_stats["stddev_ms"] = 0.0

        # 4. Build Result
        result = DecisionBenchmarkResult(
            suite_id=suite.suite_id,
            metrics=metrics,
            latency_stats=latency_stats,
            success=True,
            metadata={
                "item_count": len(raw_output.item_ids),
                "dataset_id": suite.dataset.dataset_id,
            },
        )

        # 5. Render Report
        timestamp_str = datetime.now(UTC).isoformat()
        return DecisionBenchmarkReport(
            suite_id=suite.suite_id,
            result=result,
            profile_id=profile.profile_id,
            timestamp=timestamp_str,
        )
