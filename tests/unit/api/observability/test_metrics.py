"""Unit tests for MetricsAggregator."""

from src.api.observability.collector import TelemetryCollector
from src.api.observability.metrics import MetricsAggregator
from src.api.observability.telemetry_models import ObservationEvent


class MockTimeProvider:
    def __init__(self) -> None:
        self.time_ns = 1000000000

    def now_ns(self) -> int:
        return self.time_ns


def test_metrics_aggregator_empty() -> None:
    tp = MockTimeProvider()
    aggregator = MetricsAggregator(time_provider=tp)
    metrics = aggregator.compute_metrics()
    assert metrics.total_requests == 0
    assert metrics.error_rate == 0.0


def test_metrics_aggregator_calculation() -> None:
    tp = MockTimeProvider()
    collector = TelemetryCollector(time_provider=tp)
    aggregator = MetricsAggregator(time_provider=tp)

    # 3 events: 200 (10ms), 200 (20ms), 500 (30ms)
    events = [
        ObservationEvent(
            event_id="e1",
            timestamp_ns=1000,
            correlation_id="c1",
            route_path="/a",
            http_method="GET",
            status_code=200,
            duration_ns=10_000_000,  # 10 ms
        ),
        ObservationEvent(
            event_id="e2",
            timestamp_ns=2000,
            correlation_id="c2",
            route_path="/a",
            http_method="GET",
            status_code=200,
            duration_ns=20_000_000,  # 20 ms
        ),
        ObservationEvent(
            event_id="e3",
            timestamp_ns=3000,
            correlation_id="c3",
            route_path="/b",
            http_method="POST",
            status_code=500,
            duration_ns=30_000_000,  # 30 ms
        ),
    ]

    for obs in events:
        res = collector.collect(obs)
        assert res.collected_event is not None
        aggregator.record(res.collected_event)

    metrics = aggregator.compute_metrics()
    assert metrics.total_requests == 3
    assert metrics.successful_requests == 2
    assert metrics.failed_requests == 1
    assert metrics.error_rate == round(1 / 3, 4)
    assert metrics.avg_latency_ms == round(20.0, 2)
    assert metrics.status_code_counts[200] == 2
    assert metrics.status_code_counts[500] == 1
