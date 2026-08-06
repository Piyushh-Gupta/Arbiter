"""SnapshotGenerator implementation for API Observability."""

import uuid

from src.api.observability.base import TimeProvider
from src.api.observability.telemetry_models import (
    ApiOperationalSnapshot,
    RequestMetrics,
)


class SnapshotGenerator:
    """Generates immutable ApiOperationalSnapshot objects from RequestMetrics.

    Responsibilities:
    - Consume aggregated metrics
    - Evaluate overall system health status
    - Stamp snapshot with timestamp from TimeProvider
    - Produce immutable ApiOperationalSnapshot model
    """

    def __init__(self, time_provider: TimeProvider) -> None:
        self._time_provider = time_provider

    def generate_snapshot(
        self,
        metrics: RequestMetrics,
        profile_id: str,
        active_collectors: tuple[str, ...] = ("default_collector",),
    ) -> ApiOperationalSnapshot:
        """Generates an immutable operational snapshot for a given metrics summary."""
        now_ns = self._time_provider.now_ns()
        snapshot_id = f"snap-{uuid.uuid4().hex[:12]}"

        # Determine health status based on error rate
        if metrics.error_rate >= 0.5:
            system_status = "UNHEALTHY"
        elif metrics.error_rate >= 0.1 or metrics.p95_latency_ms > 2000.0:
            system_status = "DEGRADED"
        else:
            system_status = "HEALTHY"

        return ApiOperationalSnapshot(
            snapshot_id=snapshot_id,
            timestamp_ns=now_ns,
            active_profile_id=profile_id,
            metrics=metrics,
            system_status=system_status,
            active_collectors=active_collectors,
            metadata={
                "snapshot_generator": "SnapshotGenerator",
                "version": "0.1.0",
            },
        )
