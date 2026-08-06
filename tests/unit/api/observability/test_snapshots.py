"""Unit tests for SnapshotGenerator."""

from src.api.observability.snapshots import SnapshotGenerator
from src.api.observability.telemetry_models import RequestMetrics


class MockTimeProvider:
    def now_ns(self) -> int:
        return 9000000000


def test_snapshot_generator_healthy() -> None:
    gen = SnapshotGenerator(time_provider=MockTimeProvider())
    metrics = RequestMetrics(
        total_requests=10,
        successful_requests=10,
        failed_requests=0,
        error_rate=0.0,
        p95_latency_ms=50.0,
    )
    snap = gen.generate_snapshot(metrics=metrics, profile_id="test_profile")

    assert snap.snapshot_id.startswith("snap-")
    assert snap.timestamp_ns == 9000000000
    assert snap.active_profile_id == "test_profile"
    assert snap.system_status == "HEALTHY"


def test_snapshot_generator_unhealthy() -> None:
    gen = SnapshotGenerator(time_provider=MockTimeProvider())
    metrics = RequestMetrics(
        total_requests=10,
        successful_requests=4,
        failed_requests=6,
        error_rate=0.6,
        p95_latency_ms=100.0,
    )
    snap = gen.generate_snapshot(metrics=metrics, profile_id="test_profile")
    assert snap.system_status == "UNHEALTHY"


def test_snapshot_generator_degraded() -> None:
    gen = SnapshotGenerator(time_provider=MockTimeProvider())
    metrics = RequestMetrics(
        total_requests=10,
        successful_requests=8,
        failed_requests=2,
        error_rate=0.2,
        p95_latency_ms=100.0,
    )
    snap = gen.generate_snapshot(metrics=metrics, profile_id="test_profile")
    assert snap.system_status == "DEGRADED"
