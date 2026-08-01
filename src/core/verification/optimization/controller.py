"""Verification production optimization controller and execution coordinator."""

import sys
import time
from datetime import datetime
from typing import Any

from src.core.exceptions import (
    OptimizationConfigurationError,
    OptimizationExecutionError,
    OptimizationTimeoutError,
)
from src.core.verification.optimization.implementations import (
    VerificationConcurrencyLimiter,
    VerificationTelemetryCollector,
)
from src.core.verification.optimization.optimization_models import (
    VerificationExecutionMetrics,
    VerificationOptimizationDefinition,
    VerificationOptimizationTrace,
)


class VerificationOptimizationController:
    """Coordinates verification pipeline optimization rules, concurrency bounds, and telemetry metrics."""

    def __init__(
        self,
        definition: VerificationOptimizationDefinition,
        concurrency_limiter: VerificationConcurrencyLimiter,
        telemetry_collector: VerificationTelemetryCollector | None = None,
        profile_id: str = "default_optimization",
    ) -> None:
        if not isinstance(definition, VerificationOptimizationDefinition):
            raise OptimizationConfigurationError(
                "Requires valid VerificationOptimizationDefinition."
            )
        self._definition = definition
        self._limiter = concurrency_limiter
        self._telemetry = telemetry_collector
        self._profile_id = profile_id

    @property
    def definition(self) -> VerificationOptimizationDefinition:
        return self._definition

    @property
    def concurrency_limiter(self) -> VerificationConcurrencyLimiter:
        return self._limiter

    @property
    def telemetry_collector(self) -> VerificationTelemetryCollector | None:
        return self._telemetry

    def execute(
        self,
        claim: str,
        evidence_bundle: Any,
        verifier: Any,
        verification_definition: Any,
        calibration_strategy: Any,
        calibration_definition: Any,
        explanation_strategy: Any,
        explanation_definition: Any,
    ) -> tuple[Any, VerificationExecutionMetrics, VerificationOptimizationTrace]:
        t0_total = time.perf_counter()

        # Enforce timeout bounds during semaphore acquisition
        timeout_ms = self._definition.request_timeout_ms
        acquired = self._limiter.acquire(timeout_ms=timeout_ms)
        if not acquired:
            raise OptimizationTimeoutError(
                f"Execution timed out waiting for concurrency slot after {timeout_ms}ms."
            )

        try:
            # Stage 1: Verification
            t0_ver = time.perf_counter()
            if hasattr(verifier, "verify"):
                verify_func = getattr(verifier, "verify")
                verification_result = verify_func(
                    claim, evidence_bundle, verification_definition
                )
            else:
                raise OptimizationExecutionError(
                    "Verifier must implement a verify method."
                )
            t1_ver = time.perf_counter()
            ver_ms = (t1_ver - t0_ver) * 1000.0

            # Stage 2: Calibration
            t0_cal = time.perf_counter()
            if hasattr(calibration_strategy, "calibrate"):
                calibrate_func = getattr(calibration_strategy, "calibrate")
                calibration_result = calibrate_func(
                    verification_result, calibration_definition
                )
            else:
                raise OptimizationExecutionError(
                    "Calibration strategy must implement a calibrate method."
                )
            t1_cal = time.perf_counter()
            cal_ms = (t1_cal - t0_cal) * 1000.0

            # Stage 3: Explanation
            t0_exp = time.perf_counter()
            if hasattr(explanation_strategy, "explain"):
                explain_func = getattr(explanation_strategy, "explain")
                aggregation_trace = getattr(
                    verification_result, "aggregation_trace", None
                )
                explanation_result = explain_func(
                    verification_result,
                    calibration_result,
                    evidence_bundle,
                    aggregation_trace,
                    explanation_definition,
                )
            else:
                raise OptimizationExecutionError(
                    "Explanation strategy must implement an explain method."
                )
            t1_exp = time.perf_counter()
            exp_ms = (t1_exp - t0_exp) * 1000.0

            t1_total = time.perf_counter()
            total_ms = (t1_total - t0_total) * 1000.0

            if total_ms > timeout_ms:
                raise OptimizationTimeoutError(
                    f"Execution exceeded timeout policy ({total_ms:.2f}ms > {timeout_ms}ms)."
                )

            # Gather runtime metrics
            active_slots = getattr(self._limiter, "active_slots", 0)
            batch_config = {
                "verifier": self._definition.verifier_batch_size,
                "aggregation": self._definition.aggregation_batch_size,
                "calibration": self._definition.calibration_batch_size,
                "explanation": self._definition.explanation_batch_size,
            }

            metrics = VerificationExecutionMetrics(
                verification_latency_ms=ver_ms,
                aggregation_latency_ms=0.05,  # mock/fixed aggregation latency (M2.4 has no runtime model)
                calibration_latency_ms=cal_ms,
                explanation_latency_ms=exp_ms,
                total_latency_ms=total_ms,
                throughput_qps=1000.0 / total_ms if total_ms > 0 else 0.0,
                memory_usage_bytes=sys.getsizeof(explanation_result),
                batch_sizes=batch_config,
                concurrency_active_requests=self._definition.max_concurrent_requests
                - active_slots,
            )

            if self._telemetry is not None and self._definition.telemetry_enabled:
                self._telemetry.record_execution(metrics)

            trace = VerificationOptimizationTrace(
                profile_id=self._profile_id,
                semaphore_active_slots=active_slots,
                timeout_ms=timeout_ms,
                batch_configuration=batch_config,
                telemetry_configured=self._definition.telemetry_enabled,
                execution_timestamp=datetime.utcnow().isoformat(),
            )

            return explanation_result, metrics, trace

        finally:
            self._limiter.release()
