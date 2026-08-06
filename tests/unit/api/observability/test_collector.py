"""Unit tests for TelemetryCollector."""

from src.api.observability.collector import TelemetryCollector
from src.api.observability.telemetry_models import ObservationEvent


class MockTimeProvider:
    def __init__(self, time_ns: int = 1000000000) -> None:
        self.time_ns = time_ns

    def now_ns(self) -> int:
        return self.time_ns


def test_telemetry_collector_success() -> None:
    tp = MockTimeProvider(time_ns=2000000000)
    collector = TelemetryCollector(time_provider=tp)

    obs = ObservationEvent(
        event_id="evt-123",
        timestamp_ns=1000000000,
        correlation_id="corr-abc",
        route_path="/health",
        http_method="GET",
        status_code=200,
        duration_ns=5000000,
    )

    result = collector.collect(obs)
    assert result.success is True
    assert result.event_id == "evt-123"
    assert result.collected_event is not None
    assert result.collected_event.collected_at_ns == 2000000000
    assert "method:GET" in result.collected_event.tags
    assert "status:200" in result.collected_event.tags


def test_telemetry_collector_error_isolation() -> None:
    class FaultyTimeProvider:
        def now_ns(self) -> int:
            raise RuntimeError("Clock malfunction")

    collector = TelemetryCollector(time_provider=FaultyTimeProvider())

    obs = ObservationEvent(
        event_id="evt-err",
        timestamp_ns=1000,
        correlation_id="corr-err",
        route_path="/test",
        http_method="POST",
        status_code=500,
        duration_ns=100,
    )

    result = collector.collect(obs)
    assert result.success is False
    assert result.event_id == "evt-err"
    assert result.collected_event is None
    assert "Clock malfunction" in (result.error_message or "")
