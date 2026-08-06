"""Unit tests for ObservabilityPipeline."""

from src.api.observability.collector import TelemetryCollector
from src.api.observability.metrics import MetricsAggregator
from src.api.observability.pipeline import ObservabilityPipeline
from src.api.observability.snapshots import SnapshotGenerator
from src.api.observability.telemetry_models import (
    ObservationEvent,
    TelemetryCollectionResult,
)
from src.api.observability.tracing import TracingProvider


class MockTimeProvider:
    def now_ns(self) -> int:
        return 1234567890


def test_observability_pipeline_execution() -> None:
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
        profile_id="test-p",
    )

    obs = ObservationEvent(
        event_id="e1",
        timestamp_ns=1000,
        correlation_id="c1",
        route_path="/health",
        http_method="GET",
        status_code=200,
        duration_ns=2000000,
    )

    pipeline.execute(obs)

    m = pipeline.get_metrics()
    assert m.total_requests == 1

    t = pipeline.get_latest_trace()
    assert t is not None
    assert t.correlation_id == "c1"

    s = pipeline.get_latest_snapshot()
    assert s is not None
    assert s.active_profile_id == "test-p"


def test_observability_pipeline_error_isolation() -> None:
    class FaultyCollector(TelemetryCollector):
        def collect(self, event: ObservationEvent) -> TelemetryCollectionResult:
            raise RuntimeError("Boom inside collector")

    tp = MockTimeProvider()
    collector = FaultyCollector(time_provider=tp)
    tracing = TracingProvider(time_provider=tp)
    metrics_agg = MetricsAggregator(time_provider=tp)
    snapshots = SnapshotGenerator(time_provider=tp)

    pipeline = ObservabilityPipeline(
        collector=collector,
        tracing_provider=tracing,
        metrics_aggregator=metrics_agg,
        snapshot_generator=snapshots,
    )

    obs = ObservationEvent(
        event_id="e1",
        timestamp_ns=1000,
        correlation_id="c1",
        route_path="/test",
        http_method="POST",
        status_code=500,
        duration_ns=100,
    )

    # Execution must NOT raise an exception
    pipeline.execute(obs)
    assert pipeline.get_metrics().total_requests == 0
