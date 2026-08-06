"""MetricsAggregator implementation for API Observability."""

import math
from typing import Sequence

from src.api.observability.base import TimeProvider
from src.api.observability.telemetry_models import ApiTelemetryEvent, RequestMetrics


class MetricsAggregator:
    """Consumes ApiTelemetryEvent instances and produces immutable RequestMetrics.

    Responsibilities:
    - Consume telemetry events
    - Compute request counts, status code breakdown, error rate
    - Compute latency statistics (avg, p95, p99)
    - Calculate throughput (requests per second)
    - Produce immutable RequestMetrics models
    """

    def __init__(self, time_provider: TimeProvider) -> None:
        self._time_provider = time_provider
        self._events: list[ApiTelemetryEvent] = []
        self._first_seen_ns: int | None = None

    def record(self, telemetry_event: ApiTelemetryEvent) -> None:
        """Records a telemetry event in the internal history accumulator."""
        if self._first_seen_ns is None:
            self._first_seen_ns = telemetry_event.collected_at_ns
        self._events.append(telemetry_event)

    def compute_metrics(self) -> RequestMetrics:
        """Computes and returns an immutable RequestMetrics object based on recorded events."""
        if not self._events:
            return RequestMetrics()

        total_requests = len(self._events)
        status_code_counts: dict[int, int] = {}
        successful_requests = 0
        failed_requests = 0
        latencies_ms: list[float] = []

        for event in self._events:
            obs = event.observation_event
            code = obs.status_code
            status_code_counts[code] = status_code_counts.get(code, 0) + 1

            if 200 <= code < 400:
                successful_requests += 1
            else:
                failed_requests += 1

            # Convert nanoseconds to milliseconds
            latency_ms = obs.duration_ns / 1_000_000.0
            latencies_ms.append(latency_ms)

        error_rate = failed_requests / total_requests if total_requests > 0 else 0.0
        avg_latency_ms = (
            sum(latencies_ms) / total_requests if total_requests > 0 else 0.0
        )

        latencies_sorted = sorted(latencies_ms)
        p95_latency_ms = self._percentile(latencies_sorted, 0.95)
        p99_latency_ms = self._percentile(latencies_sorted, 0.99)

        # Calculate throughput based on elapsed time
        now_ns = self._time_provider.now_ns()
        start_ns = self._first_seen_ns or now_ns
        elapsed_seconds = (now_ns - start_ns) / 1_000_000_000.0
        requests_per_second = (
            total_requests / elapsed_seconds if elapsed_seconds > 0.001 else 0.0
        )

        return RequestMetrics(
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            error_rate=round(error_rate, 4),
            avg_latency_ms=round(avg_latency_ms, 2),
            p95_latency_ms=round(p95_latency_ms, 2),
            p99_latency_ms=round(p99_latency_ms, 2),
            requests_per_second=round(requests_per_second, 2),
            status_code_counts=status_code_counts,
        )

    def _percentile(self, sorted_data: Sequence[float], percentile: float) -> float:
        """Helper to calculate percentile on sorted float data."""
        if not sorted_data:
            return 0.0
        n = len(sorted_data)
        if n == 1:
            return sorted_data[0]
        k = (n - 1) * percentile
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_data[int(k)]
        d0 = sorted_data[int(f)] * (c - k)
        d1 = sorted_data[int(c)] * (k - f)
        return d0 + d1
