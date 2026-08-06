"""Base protocols for API Observability, Telemetry & Monitoring."""

import time
from typing import Protocol, runtime_checkable

from src.api.observability.telemetry_models import (
    ApiOperationalSnapshot,
    ApiTelemetryEvent,
    ObservationEvent,
    RequestMetrics,
    RequestTrace,
    TelemetryCollectionResult,
)


@runtime_checkable
class TimeProvider(Protocol):
    """Protocol for a deterministic clock / time provider abstraction."""

    def now_ns(self) -> int:
        """Returns current timestamp in nanoseconds."""
        ...


class SystemTimeProvider:
    """Default system clock time provider implementation."""

    def now_ns(self) -> int:
        """Returns the current system time in nanoseconds."""
        return time.time_ns()


@runtime_checkable
class BaseTelemetryCollector(Protocol):
    """Protocol for stateless/deterministic telemetry collection."""

    def collect(self, event: ObservationEvent) -> TelemetryCollectionResult:
        """Produces an immutable ApiTelemetryEvent from an ObservationEvent."""
        ...


@runtime_checkable
class BaseTracingProvider(Protocol):
    """Protocol for distributed tracing creation and context propagation."""

    def create_trace(self, event: ObservationEvent) -> RequestTrace:
        """Creates an immutable RequestTrace for an ObservationEvent."""
        ...

    def extract_trace_context(self, headers: dict[str, str]) -> dict[str, str]:
        """Extracts trace context from HTTP headers."""
        ...

    def inject_trace_context(self, trace: RequestTrace) -> dict[str, str]:
        """Injects trace context into HTTP response headers."""
        ...


@runtime_checkable
class BaseMetricsAggregator(Protocol):
    """Protocol for accumulating telemetry events into immutable RequestMetrics."""

    def record(self, telemetry_event: ApiTelemetryEvent) -> None:
        """Records a telemetry event."""
        ...

    def compute_metrics(self) -> RequestMetrics:
        """Computes and returns immutable aggregated metrics."""
        ...


@runtime_checkable
class BaseSnapshotGenerator(Protocol):
    """Protocol for generating operational system snapshots."""

    def generate_snapshot(
        self, metrics: RequestMetrics, profile_id: str
    ) -> ApiOperationalSnapshot:
        """Generates an immutable operational snapshot."""
        ...


@runtime_checkable
class BaseMonitoringService(Protocol):
    """Protocol for standard monitoring orchestration."""

    def process_observation(self, event: ObservationEvent) -> None:
        """Processes an ObservationEvent through the observability pipeline."""
        ...

    def get_current_metrics(self) -> RequestMetrics:
        """Returns the current aggregated request metrics."""
        ...

    def get_latest_snapshot(self) -> ApiOperationalSnapshot | None:
        """Returns the latest operational snapshot if generated."""
        ...
