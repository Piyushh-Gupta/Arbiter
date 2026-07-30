"""Thread-safe operational telemetry collector for production retrieval optimization."""

import threading
import time

from src.core.retrieval.optimization.optimization_models import (
    RetrievalExecutionMetrics,
    TelemetrySnapshot,
)


class TelemetryCollector:
    """
    Thread-safe observational telemetry aggregator.
    Aggregates per-request execution metrics without influencing execution.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._start_time = time.perf_counter()
        self._total_requests = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._latencies_ms: list[float] = []

    def record_execution(self, metrics: RetrievalExecutionMetrics) -> None:
        with self._lock:
            self._total_requests += 1
            if metrics.cache_hit:
                self._cache_hits += 1
            else:
                self._cache_misses += 1
            self._latencies_ms.append(metrics.total_latency_ms)

    def snapshot(self) -> TelemetrySnapshot:
        with self._lock:
            total = self._total_requests
            hits = self._cache_hits
            misses = self._cache_misses
            latencies = list(self._latencies_ms)
            elapsed_sec = max(0.001, time.perf_counter() - self._start_time)

        if not latencies:
            return TelemetrySnapshot(
                total_requests=total,
                cache_hits=hits,
                cache_misses=misses,
                average_latency_ms=0.0,
                p95_latency_ms=0.0,
                throughput_qps=0.0,
            )

        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)
        avg_lat = float(sum(sorted_latencies) / n)
        p95_idx = min(n - 1, int(n * 0.95))
        p95_lat = float(sorted_latencies[p95_idx])
        qps = float(total / elapsed_sec)

        return TelemetrySnapshot(
            total_requests=total,
            cache_hits=hits,
            cache_misses=misses,
            average_latency_ms=avg_lat,
            p95_latency_ms=p95_lat,
            throughput_qps=qps,
        )

    def clear(self) -> None:
        with self._lock:
            self._start_time = time.perf_counter()
            self._total_requests = 0
            self._cache_hits = 0
            self._cache_misses = 0
            self._latencies_ms.clear()
