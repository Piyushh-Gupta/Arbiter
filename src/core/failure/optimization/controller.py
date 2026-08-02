"""Failure optimization controller and execution coordinator (M3.8)."""

import time
import uuid
from datetime import datetime, timezone
from typing import Any

from src.core.exceptions import OptimizationConfigurationError, OptimizationTimeoutError
from src.core.failure.failure_models import FailureAnalysisInput
from src.core.failure.optimization.implementations import (
    BoundedSemaphoreConcurrencyLimiter,
    FailureTelemetryCollector,
)
from src.core.failure.optimization.optimization_models import (
    FailureExecutionMetrics,
    FailureOptimizationDefinition,
    FailureTelemetryRecord,
)


class FailureOptimizationController:
    """
    Stateless controller coordinating failure analysis execution under bounded concurrency,
    timeouts, and stage-by-stage latency tracking.
    """

    def __init__(
        self,
        definition: FailureOptimizationDefinition,
        concurrency_limiter: BoundedSemaphoreConcurrencyLimiter,
        telemetry_collector: FailureTelemetryCollector | None = None,
    ) -> None:
        if not isinstance(definition, FailureOptimizationDefinition):
            raise OptimizationConfigurationError(
                "Requires valid FailureOptimizationDefinition."
            )
        if not isinstance(concurrency_limiter, BoundedSemaphoreConcurrencyLimiter):
            raise OptimizationConfigurationError(
                "Requires valid BoundedSemaphoreConcurrencyLimiter."
            )
        self._definition = definition
        self._limiter = concurrency_limiter
        self._telemetry = telemetry_collector

    @property
    def definition(self) -> FailureOptimizationDefinition:
        return self._definition

    @property
    def concurrency_limiter(self) -> BoundedSemaphoreConcurrencyLimiter:
        return self._limiter

    @property
    def telemetry_collector(self) -> FailureTelemetryCollector | None:
        return self._telemetry

    def validate_compatibility(self, definition: FailureOptimizationDefinition) -> None:
        """Statically verifies compatibility of optimization configuration settings."""
        if not isinstance(definition, FailureOptimizationDefinition):
            raise OptimizationConfigurationError(
                "Invalid definition type for FailureOptimizationController."
            )

    def execute(
        self,
        input_data: FailureAnalysisInput,
        analyzer: Any,
        correlation_engine: Any | None = None,
        attribution_engine: Any | None = None,
        severity_policy: Any | None = None,
        explainer: Any | None = None,
    ) -> tuple[Any, FailureExecutionMetrics]:
        """
        Executes failure analysis stages under concurrency limits, measuring per-stage latencies
        and emitting telemetry records.
        """
        request_id = str(uuid.uuid4())
        t0_total = time.perf_counter()

        acquired = self._limiter.acquire(timeout_ms=self._definition.timeout_ms)
        if not acquired:
            raise OptimizationTimeoutError(
                f"Execution timed out waiting for concurrency slot after {self._definition.timeout_ms}ms."
            )

        success = True
        error_msg: str | None = None
        analysis_res: Any = None
        corr_res: Any = None
        attr_res: Any = None
        sev_res: Any = None
        expl_res: Any = None

        t_anal_ms = 0.0
        t_corr_ms = 0.0
        t_attr_ms = 0.0
        t_sev_ms = 0.0
        t_expl_ms = 0.0

        try:
            # Stage 1: Failure Analysis
            t0 = time.perf_counter()
            if hasattr(analyzer, "analyze"):
                analysis_res = analyzer.analyze(input_data)
            elif callable(analyzer):
                analysis_res = analyzer(input_data)
            t_anal_ms = (time.perf_counter() - t0) * 1000.0

            # Stage 2: Correlation Engine (optional)
            if correlation_engine is not None:
                t0 = time.perf_counter()
                if hasattr(correlation_engine, "correlate"):
                    corr_res = correlation_engine.correlate(analysis_res)
                elif callable(correlation_engine):
                    corr_res = correlation_engine(analysis_res)
                t_corr_ms = (time.perf_counter() - t0) * 1000.0

            # Stage 3: Attribution Engine (optional)
            if attribution_engine is not None:
                t0 = time.perf_counter()
                if hasattr(attribution_engine, "attribute"):
                    attr_res = attribution_engine.attribute(corr_res or analysis_res)
                elif callable(attribution_engine):
                    attr_res = attribution_engine(corr_res or analysis_res)
                t_attr_ms = (time.perf_counter() - t0) * 1000.0

            # Stage 4: Severity Policy (optional)
            if severity_policy is not None:
                t0 = time.perf_counter()
                if hasattr(severity_policy, "evaluate"):
                    sev_res = severity_policy.evaluate(
                        attr_res or corr_res or analysis_res
                    )
                elif callable(severity_policy):
                    sev_res = severity_policy(attr_res or corr_res or analysis_res)
                t_sev_ms = (time.perf_counter() - t0) * 1000.0

            # Stage 5: Explainability Strategy (optional)
            if explainer is not None:
                t0 = time.perf_counter()
                if hasattr(explainer, "explain"):
                    expl_res = explainer.explain(
                        analysis_res, corr_res, attr_res, sev_res
                    )
                elif callable(explainer):
                    expl_res = explainer(analysis_res, corr_res, attr_res, sev_res)
                t_expl_ms = (time.perf_counter() - t0) * 1000.0

        except Exception as e:
            success = False
            error_msg = str(e)
            raise e
        finally:
            self._limiter.release()
            t_total_ms = (time.perf_counter() - t0_total) * 1000.0

            metrics = FailureExecutionMetrics(
                analysis_latency_ms=t_anal_ms,
                correlation_latency_ms=t_corr_ms,
                attribution_latency_ms=t_attr_ms,
                severity_latency_ms=t_sev_ms,
                explainability_latency_ms=t_expl_ms,
                total_latency_ms=t_total_ms,
                memory_usage_mb=0.0,
                analyzer_count=1,
            )

            if self._telemetry is not None and self._definition.telemetry_enabled:
                rec = FailureTelemetryRecord(
                    request_id=request_id,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    execution_metrics=metrics,
                    success=success,
                    error_message=error_msg,
                )
                self._telemetry.record(rec)

        final_result = expl_res or sev_res or attr_res or corr_res or analysis_res
        return final_result, metrics
