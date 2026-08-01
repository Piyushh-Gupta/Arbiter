"""Generic concurrency and telemetry implementations for Verification Production Optimization."""

import threading
import time
from typing import Any, Protocol, runtime_checkable

from src.core.exceptions import OptimizationTimeoutError
from src.core.verification.optimization.optimization_models import (
    VerificationExecutionMetrics,
    VerificationTelemetrySnapshot,
)


@runtime_checkable
class VerificationConcurrencyLimiter(Protocol):
    """Protocol abstracting synchronization bounds and concurrency control."""

    def acquire(self, timeout_ms: float | None = None) -> bool:
        """Acquires a slot within the specified timeout in milliseconds."""
        ...

    def release(self) -> None:
        """Releases the acquired slot."""
        ...

    def __enter__(self) -> "VerificationConcurrencyLimiter":
        """Context manager entry acquiring slot."""
        ...

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit releasing slot."""
        ...


class BoundedSemaphoreVerificationConcurrencyLimiter(VerificationConcurrencyLimiter):
    """Generic BoundedSemaphore implementation ensuring strict request concurrency boundaries."""

    def __init__(self, max_concurrency: int) -> None:
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive.")
        self._max_concurrency = max_concurrency
        self._semaphore = threading.BoundedSemaphore(max_concurrency)
        self._active_slots = max_concurrency

    @property
    def max_concurrency(self) -> int:
        return self._max_concurrency

    @property
    def active_slots(self) -> int:
        return self._active_slots

    def acquire(self, timeout_ms: float | None = None) -> bool:
        if timeout_ms is not None:
            timeout_sec = max(0.0, timeout_ms / 1000.0)
            acquired = self._semaphore.acquire(timeout=timeout_sec)
        else:
            acquired = self._semaphore.acquire()

        if acquired:
            self._active_slots -= 1
        return acquired

    def release(self) -> None:
        try:
            self._semaphore.release()
            self._active_slots = min(self._max_concurrency, self._active_slots + 1)
        except ValueError as e:
            raise ValueError("Released more times than acquired.") from e

    def __enter__(self) -> "BoundedSemaphoreVerificationConcurrencyLimiter":
        if not self.acquire():
            raise OptimizationTimeoutError("Failed to acquire concurrency lock.")
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()


class VerificationTelemetryCollector:
    """Thread-safe telemetry collector to observe verification pipeline performance."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._start_time = time.perf_counter()
        self._total_requests = 0
        self._latencies_ms: list[float] = []
        self._peak_concurrency = 0

    def record_execution(self, metrics: VerificationExecutionMetrics) -> None:
        with self._lock:
            self._total_requests += 1
            self._latencies_ms.append(metrics.total_latency_ms)
            if metrics.concurrency_active_requests > self._peak_concurrency:
                self._peak_concurrency = metrics.concurrency_active_requests

    def snapshot(self) -> VerificationTelemetrySnapshot:
        with self._lock:
            total = self._total_requests
            latencies = list(self._latencies_ms)
            peak = self._peak_concurrency
            elapsed = max(0.001, time.perf_counter() - self._start_time)

        if not latencies:
            return VerificationTelemetrySnapshot(
                total_requests=total,
                average_latency_ms=0.0,
                p95_latency_ms=0.0,
                throughput_qps=0.0,
                peak_concurrency=peak,
            )

        sorted_lats = sorted(latencies)
        n = len(sorted_lats)
        avg = sum(sorted_lats) / n
        p95_idx = min(n - 1, int(n * 0.95))
        p95 = sorted_lats[p95_idx]
        qps = total / elapsed

        return VerificationTelemetrySnapshot(
            total_requests=total,
            average_latency_ms=float(avg),
            p95_latency_ms=float(p95),
            throughput_qps=float(qps),
            peak_concurrency=peak,
        )

    def clear(self) -> None:
        with self._lock:
            self._start_time = time.perf_counter()
            self._total_requests = 0
            self._latencies_ms.clear()
            self._peak_concurrency = 0
