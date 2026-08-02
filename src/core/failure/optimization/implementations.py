"""Concurrency limiters and telemetry collectors for Failure Analysis Optimization (M3.8)."""

import math
import threading
import time
from typing import Sequence

from src.core.failure.optimization.optimization_models import (
    FailureTelemetryRecord,
    FailureTelemetrySnapshot,
)


class BoundedSemaphoreConcurrencyLimiter:
    """Thread-safe concurrency slot isolation bounded by a max capacity and timeout."""

    def __init__(self, max_concurrent_requests: int = 4) -> None:
        if max_concurrent_requests <= 0:
            raise ValueError("max_concurrent_requests must be positive.")
        self._max_capacity = max_concurrent_requests
        self._semaphore = threading.BoundedSemaphore(value=max_concurrent_requests)
        self._lock = threading.Lock()
        self._active_count = 0

    @property
    def max_capacity(self) -> int:
        return self._max_capacity

    @property
    def active_slots(self) -> int:
        with self._lock:
            return self._active_count

    def acquire(self, timeout_ms: float = 5000.0) -> bool:
        """Acquires a slot within timeout_ms. Returns True if acquired, False on timeout."""
        timeout_sec = max(0.0, timeout_ms / 1000.0)
        acquired = self._semaphore.acquire(timeout=timeout_sec)
        if acquired:
            with self._lock:
                self._active_count += 1
        return acquired

    def release(self) -> None:
        """Releases an acquired concurrency slot."""
        with self._lock:
            if self._active_count > 0:
                self._active_count -= 1
        self._semaphore.release()


class FailureTelemetryCollector:
    """Thread-safe collector consuming FailureTelemetryRecord items and aggregating FailureTelemetrySnapshot."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: list[FailureTelemetryRecord] = []
        self._start_time = time.time()

    def record(self, telemetry_record: FailureTelemetryRecord) -> None:
        """Consumes a FailureTelemetryRecord into operational history."""
        with self._lock:
            self._records.append(telemetry_record)

    def get_records(self) -> tuple[FailureTelemetryRecord, ...]:
        """Returns all recorded telemetry records."""
        with self._lock:
            return tuple(self._records)

    def snapshot(self) -> FailureTelemetrySnapshot:
        """Aggregates recorded telemetry into an immutable FailureTelemetrySnapshot."""
        with self._lock:
            if not self._records:
                return FailureTelemetrySnapshot()

            total_requests = len(self._records)
            failure_count = sum(1 for r in self._records if not r.success)
            latencies = sorted(
                r.execution_metrics.total_latency_ms for r in self._records
            )

            avg_latency = sum(latencies) / total_requests

            def _percentile(data: Sequence[float], p: float) -> float:
                if not data:
                    return 0.0
                k = (len(data) - 1) * p
                f = math.floor(k)
                c = math.ceil(k)
                if f == c:
                    return data[int(k)]
                d0 = data[int(f)] * (c - k)
                d1 = data[int(c)] * (k - f)
                return d0 + d1

            p95 = _percentile(latencies, 0.95)
            p99 = _percentile(latencies, 0.99)

            elapsed = max(0.001, time.time() - self._start_time)
            throughput = total_requests / elapsed

            return FailureTelemetrySnapshot(
                total_requests=total_requests,
                average_latency_ms=avg_latency,
                p95_latency_ms=p95,
                p99_latency_ms=p99,
                throughput_qps=throughput,
                failure_count=failure_count,
            )
