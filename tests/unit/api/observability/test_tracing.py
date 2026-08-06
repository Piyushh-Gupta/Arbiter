"""Unit tests for TracingProvider."""

from src.api.observability.telemetry_models import ObservationEvent
from src.api.observability.tracing import TracingProvider


class MockTimeProvider:
    def now_ns(self) -> int:
        return 5000000000


def test_tracing_provider_create_trace() -> None:
    provider = TracingProvider(time_provider=MockTimeProvider())
    obs = ObservationEvent(
        event_id="evt-tr",
        timestamp_ns=1000000000,
        correlation_id="corr-xyz",
        route_path="/v1/evaluate",
        http_method="POST",
        status_code=200,
        duration_ns=10000000,
        metadata={"trace_id": "custom-trace-123"},
    )

    trace = provider.create_trace(obs)
    assert trace.trace_id == "custom-trace-123"
    assert trace.correlation_id == "corr-xyz"
    assert trace.start_time_ns == 1000000000
    assert trace.end_time_ns == 1010000000
    assert trace.attributes["http.method"] == "POST"


def test_tracing_provider_extract_and_inject() -> None:
    provider = TracingProvider(time_provider=MockTimeProvider())

    headers = {"X-Trace-ID": "tr-abc-999", "X-Parent-Span-ID": "sp-parent-000"}
    extracted = provider.extract_trace_context(headers)
    assert extracted["trace_id"] == "tr-abc-999"
    assert extracted["parent_span_id"] == "sp-parent-000"

    obs = ObservationEvent(
        event_id="e1",
        timestamp_ns=100,
        correlation_id="corr-1",
        route_path="/",
        http_method="GET",
        status_code=200,
        duration_ns=50,
        metadata=extracted,
    )

    trace = provider.create_trace(obs)
    assert trace.trace_id == "tr-abc-999"
    assert trace.parent_span_id == "sp-parent-000"

    injected = provider.inject_trace_context(trace)
    assert injected["x-trace-id"] == "tr-abc-999"
    assert injected["x-correlation-id"] == "corr-1"
    assert "x-span-id" in injected
