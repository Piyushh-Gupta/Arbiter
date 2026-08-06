"""TelemetryCollector implementation for API Observability."""

import uuid

from src.api.observability.base import TimeProvider
from src.api.observability.telemetry_models import (
    ApiTelemetryEvent,
    ObservationEvent,
    TelemetryCollectionResult,
)


class TelemetryCollector:
    """Produces immutable ApiTelemetryEvent objects from ObservationEvents.

    Responsibilities:
    - Receive observation events
    - Stamp telemetry events with timestamps via TimeProvider
    - Produce immutable TelemetryCollectionResult
    - Never aggregate metrics
    """

    def __init__(self, time_provider: TimeProvider) -> None:
        self._time_provider = time_provider

    def collect(self, event: ObservationEvent) -> TelemetryCollectionResult:
        """Transforms an ObservationEvent into an immutable ApiTelemetryEvent."""
        try:
            now_ns = self._time_provider.now_ns()
            telemetry_id = f"telem-{uuid.uuid4().hex[:12]}"
            tags = (
                f"method:{event.http_method}",
                f"status:{event.status_code}",
                f"route:{event.route_path}",
            )
            telemetry_event = ApiTelemetryEvent(
                telemetry_id=telemetry_id,
                observation_event=event,
                collected_at_ns=now_ns,
                tags=tags,
            )
            return TelemetryCollectionResult(
                event_id=event.event_id,
                success=True,
                collected_event=telemetry_event,
            )
        except Exception as e:
            return TelemetryCollectionResult(
                event_id=event.event_id,
                success=False,
                error_message=str(e),
            )
