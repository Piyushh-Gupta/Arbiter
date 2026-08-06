"""Unit tests for immutable telemetry data models."""

import pytest
from pydantic import ValidationError

from src.api.observability.telemetry_models import (
    ApiOperationalSnapshot,
    ApiTelemetryEvent,
    MonitoringProfile,
    ObservationEvent,
    RequestMetrics,
    RequestTrace,
    TelemetryCollectionResult,
)


def test_observation_event_immutability() -> None:
    event = ObservationEvent(
        event_id="evt-1",
        timestamp_ns=1000,
        correlation_id="corr-1",
        route_path="/v1/evaluate",
        http_method="POST",
        status_code=200,
        duration_ns=5000000,
    )
    assert event.event_id == "evt-1"
    assert event.status_code == 200

    with pytest.raises(ValidationError):
        setattr(event, "status_code", 500)


def test_telemetry_event_immutability() -> None:
    obs = ObservationEvent(
        event_id="evt-1",
        timestamp_ns=1000,
        correlation_id="corr-1",
        route_path="/health",
        http_method="GET",
        status_code=200,
        duration_ns=1000000,
    )
    telem = ApiTelemetryEvent(
        telemetry_id="telem-1",
        observation_event=obs,
        collected_at_ns=1050,
    )
    assert telem.telemetry_id == "telem-1"

    with pytest.raises(ValidationError):
        setattr(telem, "telemetry_id", "telem-2")


def test_request_trace_immutability() -> None:
    trace = RequestTrace(
        trace_id="tr-1",
        span_id="sp-1",
        correlation_id="corr-1",
        start_time_ns=1000,
        end_time_ns=2000,
    )
    assert trace.trace_id == "tr-1"

    with pytest.raises(ValidationError):
        setattr(trace, "trace_id", "tr-2")


def test_request_metrics_immutability() -> None:
    metrics = RequestMetrics(
        total_requests=10,
        successful_requests=8,
        failed_requests=2,
        error_rate=0.2,
        avg_latency_ms=15.5,
    )
    assert metrics.error_rate == 0.2

    with pytest.raises(ValidationError):
        setattr(metrics, "total_requests", 20)


def test_api_operational_snapshot_immutability() -> None:
    metrics = RequestMetrics(total_requests=1)
    snap = ApiOperationalSnapshot(
        snapshot_id="snap-1",
        timestamp_ns=5000,
        active_profile_id="default",
        metrics=metrics,
    )
    assert snap.system_status == "HEALTHY"

    with pytest.raises(ValidationError):
        setattr(snap, "system_status", "DEGRADED")


def test_monitoring_profile_immutability() -> None:
    profile = MonitoringProfile(profile_id="p-1", snapshot_interval_seconds=30.0)
    assert profile.snapshot_interval_seconds == 30.0

    with pytest.raises(ValidationError):
        setattr(profile, "snapshot_interval_seconds", 10.0)


def test_telemetry_collection_result_immutability() -> None:
    res = TelemetryCollectionResult(event_id="e-1", success=True)
    assert res.success is True

    with pytest.raises(ValidationError):
        setattr(res, "success", False)
