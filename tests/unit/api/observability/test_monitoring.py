"""Unit tests for MonitoringService."""

from src.api.observability.collector import TelemetryCollector
from src.api.observability.metrics import MetricsAggregator
from src.api.observability.monitoring import MonitoringService
from src.api.observability.pipeline import ObservabilityPipeline
from src.api.observability.snapshots import SnapshotGenerator
from src.api.observability.telemetry_models import ObservationEvent
from src.api.observability.tracing import TracingProvider


class MockTimeProvider:
    def now_ns(self) -> int:
        return 1000000000


def test_monitoring_service_flow() -> None:
    tp = MockTimeProvider()
    collector = TelemetryCollector(time_provider=tp)
    tracing = TracingProvider(time_provider=tp)
    metrics_agg = MetricsAggregator(time_provider=tp)
    snapshots = SnapshotGenerator(time_provider=tp)

    pipeline = ObservabilityPipeline(
        collector=collector,
        tracing_provider=tracing,
        metrics_aggregator=metrics_agg,
        snapshot_generator=snapshots,
    )
    service = MonitoringService(pipeline=pipeline)

    obs = ObservationEvent(
        event_id="e-test",
        timestamp_ns=100,
        correlation_id="c-test",
        route_path="/health",
        http_method="GET",
        status_code=200,
        duration_ns=1000000,
    )

    service.process_observation(obs)

    m = service.get_current_metrics()
    assert m.total_requests == 1

    health_info = service.get_health_metrics()
    assert health_info["status"] == "HEALTHY"
    assert health_info["total_requests"] == 1
