"""Executor, Runner, and Report Builder for Pipeline Benchmarking & Evaluation Framework (M5.4)."""

import statistics
from datetime import datetime, timezone
from typing import Any, Sequence

from src.core.pipeline.benchmark.base import (
    BaseClock,
    BasePipelineBenchmarkExecutor,
    BasePipelineBenchmarkRunner,
)
from src.core.pipeline.benchmark.benchmark_models import (
    PipelineBenchmarkDataset,
    PipelineBenchmarkMetrics,
    PipelineBenchmarkProfile,
    PipelineBenchmarkRawOutput,
    PipelineBenchmarkReport,
    PipelineBenchmarkResult,
    PipelineBenchmarkSuite,
    PipelineFailureRecord,
    PipelineStageBenchmarkMetrics,
)
from src.core.pipeline.benchmark.metrics import PipelineBenchmarkMetricEngine


class SystemClock(BaseClock):
    """Monotonic system clock using time.perf_counter()."""

    def now_ms(self) -> float:
        import time

        return time.perf_counter() * 1000.0


def extract_failure_record(e: Exception) -> PipelineFailureRecord:
    """Helper to extract structured failure information from an exception."""
    exc_type = type(e).__name__
    err_msg = str(e)
    category = "UNKNOWN"

    # Match failure categories
    if "Timeout" in exc_type or "timeout" in err_msg.lower():
        category = "TIMEOUT"
    elif "Recovery" in exc_type or "recovery" in err_msg.lower():
        category = "RECOVERY_FAILED"
    elif "Stage" in exc_type or "stage" in err_msg.lower():
        category = "STAGE_ERROR"
    else:
        category = "SYSTEM_ERROR"

    attempts = 0
    res_meta = getattr(e, "resilience_metadata", None)
    if res_meta and hasattr(res_meta, "retry_trace") and res_meta.retry_trace:
        attempts = res_meta.retry_trace.total_attempts

    return PipelineFailureRecord(
        exception_type=exc_type,
        error_message=err_msg,
        failure_category=category,
        retry_attempts=attempts,
    )


class PipelineBenchmarkExecutor(BasePipelineBenchmarkExecutor):
    """Stateless executor driving the orchestrator over a dataset to generate raw metrics."""

    def __init__(self, clock: BaseClock | None = None) -> None:
        self._clock = clock or SystemClock()

    def execute(
        self,
        dataset: PipelineBenchmarkDataset,
        orchestrator: Any,
    ) -> PipelineBenchmarkRawOutput:
        """Executes the orchestrator over all dataset items, gathering latencies and failures."""
        item_ids = []
        claims = []
        expected_successes = []
        actual_successes = []
        total_latencies_ms = []
        stage_latencies_ms = []
        retry_attempt_counts = []
        timeout_triggered = []
        recovery_invoked = []
        failures: list[PipelineFailureRecord | None] = []

        # We import here to avoid circular dependency issues at the package level
        from src.core.pipeline.pipeline_models import PipelineExecutionRequest

        for item in dataset.items:
            request = PipelineExecutionRequest(
                claim=item.claim,
                pipeline_profile_id=item.pipeline_profile_id,
            )

            start = self._clock.now_ms()
            result = None
            exception_val = None

            try:
                result = orchestrator.execute(request)
            except Exception as e:
                exception_val = e

            end = self._clock.now_ms()
            latency = end - start

            item_ids.append(item.item_id)
            claims.append(item.claim)
            expected_successes.append(item.expected_success)

            if exception_val is None and result is not None:
                # Execution succeeded
                actual_successes.append(result.execution_context.success)

                # Use pipeline's internal latency metric if available, else wall-clock fallback
                e2e_latency = getattr(
                    result.execution_context, "total_latency_ms", latency
                )
                total_latencies_ms.append(float(e2e_latency))

                # Extract stage-level metadata
                stages = {}
                if (
                    hasattr(result.execution_context, "stage_metadata")
                    and result.execution_context.stage_metadata
                ):
                    for stage in result.execution_context.stage_metadata:
                        stages[stage.stage_id] = float(stage.latency_ms)
                stage_latencies_ms.append(stages)

                # Resilience metadata extraction (soft dependency)
                res_meta = getattr(result, "resilience_metadata", None)
                if res_meta:
                    attempts = getattr(res_meta.retry_trace, "total_attempts", 0)
                    to_trig = False
                    if (
                        hasattr(res_meta.retry_trace, "attempts")
                        and res_meta.retry_trace.attempts
                    ):
                        to_trig = any(
                            "Timeout" in att.error_type
                            for att in res_meta.retry_trace.attempts
                        )
                    if not to_trig and getattr(
                        res_meta.retry_trace, "terminal_error", None
                    ):
                        to_trig = "Timeout" in res_meta.retry_trace.terminal_error

                    retry_attempt_counts.append(attempts)
                    timeout_triggered.append(to_trig)
                    recovery_invoked.append(
                        getattr(res_meta, "recovery_invoked", False)
                    )
                else:
                    retry_attempt_counts.append(0)
                    timeout_triggered.append(False)
                    recovery_invoked.append(False)

                failures.append(None)
            else:
                # Execution failed
                actual_successes.append(False)
                total_latencies_ms.append(float(latency))
                stage_latencies_ms.append({})

                # Resilience metadata from exception if present
                res_meta = getattr(exception_val, "resilience_metadata", None)
                if res_meta:
                    attempts = getattr(res_meta.retry_trace, "total_attempts", 0)
                    to_trig = False
                    if (
                        hasattr(res_meta.retry_trace, "attempts")
                        and res_meta.retry_trace.attempts
                    ):
                        to_trig = any(
                            "Timeout" in att.error_type
                            for att in res_meta.retry_trace.attempts
                        )
                    if not to_trig and getattr(
                        res_meta.retry_trace, "terminal_error", None
                    ):
                        to_trig = "Timeout" in res_meta.retry_trace.terminal_error

                    retry_attempt_counts.append(attempts)
                    timeout_triggered.append(to_trig)
                    recovery_invoked.append(
                        getattr(res_meta, "recovery_invoked", False)
                    )
                else:
                    retry_attempt_counts.append(0)
                    timeout_triggered.append(False)
                    recovery_invoked.append(False)

                if exception_val is not None:
                    failures.append(extract_failure_record(exception_val))
                else:
                    failures.append(
                        PipelineFailureRecord(
                            exception_type="UnknownError",
                            error_message="Pipeline execution returned no result and no exception",
                            failure_category="UNKNOWN",
                            retry_attempts=0,
                        )
                    )

        return PipelineBenchmarkRawOutput(
            suite_id=dataset.dataset_id,
            item_ids=tuple(item_ids),
            claims=tuple(claims),
            expected_successes=tuple(expected_successes),
            actual_successes=tuple(actual_successes),
            total_latencies_ms=tuple(total_latencies_ms),
            stage_latencies_ms=tuple(stage_latencies_ms),
            retry_attempt_counts=tuple(retry_attempt_counts),
            timeout_triggered=tuple(timeout_triggered),
            recovery_invoked=tuple(recovery_invoked),
            failures=tuple(failures),
        )


class PipelineBenchmarkReportBuilder:
    """Helper to assemble immutable PipelineBenchmarkReport records."""

    @staticmethod
    def calculate_latency_stats(latencies: Sequence[float]) -> dict[str, float]:
        """Computes aggregate statistical latency values."""
        if not latencies:
            return {
                "min_ms": 0.0,
                "max_ms": 0.0,
                "mean_ms": 0.0,
                "stddev_ms": 0.0,
            }
        latency_stats = {
            "min_ms": float(min(latencies)),
            "max_ms": float(max(latencies)),
            "mean_ms": float(statistics.mean(latencies)),
        }
        if len(latencies) > 1:
            latency_stats["stddev_ms"] = float(statistics.stdev(latencies))
        else:
            latency_stats["stddev_ms"] = 0.0
        return latency_stats

    @staticmethod
    def build_report(
        suite_id: str,
        profile_id: str,
        pipeline_profile_id: str,
        raw_output: PipelineBenchmarkRawOutput,
        metrics: PipelineBenchmarkMetrics,
        stage_metrics: dict[str, PipelineStageBenchmarkMetrics],
        success: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> PipelineBenchmarkReport:
        """Assembles the final structured report."""
        latency_stats = PipelineBenchmarkReportBuilder.calculate_latency_stats(
            raw_output.total_latencies_ms
        )

        result = PipelineBenchmarkResult(
            suite_id=suite_id,
            metrics=metrics,
            stage_metrics=stage_metrics,
            latency_stats=latency_stats,
            item_count=len(raw_output.item_ids),
            success=success,
            metadata=metadata or {},
        )

        timestamp_str = datetime.now(timezone.utc).isoformat()
        return PipelineBenchmarkReport(
            suite_id=suite_id,
            result=result,
            profile_id=profile_id,
            pipeline_profile_id=pipeline_profile_id,
            timestamp=timestamp_str,
        )


class PipelineBenchmarkRunner(BasePipelineBenchmarkRunner):
    """Orchestrates offline pipeline benchmark runs."""

    def __init__(
        self,
        orchestrator: Any,
        executor: BasePipelineBenchmarkExecutor | None = None,
        metric_engine: PipelineBenchmarkMetricEngine | None = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._executor = executor or PipelineBenchmarkExecutor()
        self._metric_engine = metric_engine or PipelineBenchmarkMetricEngine()

    def run(
        self,
        suite: PipelineBenchmarkSuite,
        profile: PipelineBenchmarkProfile,
    ) -> PipelineBenchmarkReport:
        """Runs the offline benchmark suite against the orchestrator, computing metrics."""
        # 1. Drive dataset execution
        raw_output = self._executor.execute(suite.dataset, self._orchestrator)

        # 2. Compute aggregate metrics
        metrics = self._metric_engine.compute(
            raw_output, enabled_metrics=profile.definition.enabled_metrics
        )

        # 3. Compute stage metrics if breakdown enabled
        stage_metrics = {}
        if profile.definition.include_stage_breakdown:
            stage_metrics = self._metric_engine.compute_stage_metrics(raw_output)

        # 4. Resolve pipeline profile ID fallback
        pipeline_profile_id = "unknown"
        if suite.dataset.items:
            pipeline_profile_id = suite.dataset.items[0].pipeline_profile_id

        # 5. Assemble final report
        return PipelineBenchmarkReportBuilder.build_report(
            suite_id=suite.suite_id,
            profile_id=profile.profile_id,
            pipeline_profile_id=pipeline_profile_id,
            raw_output=raw_output,
            metrics=metrics,
            stage_metrics=stage_metrics,
            success=True,
            metadata={
                "dataset_id": suite.dataset.dataset_id,
                "item_count": len(raw_output.item_ids),
            },
        )
