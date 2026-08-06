"""ObservabilityPipeline implementation for API Observability."""

import logging

from src.api.observability.collector import TelemetryCollector
from src.api.observability.metrics import MetricsAggregator
from src.api.observability.snapshots import SnapshotGenerator
from src.api.observability.telemetry_models import (
    ApiOperationalSnapshot,
    ObservationEvent,
    RequestMetrics,
    RequestTrace,
)
from src.api.observability.tracing import TracingProvider

logger = logging.getLogger("arbiter.api.observability")


class ObservabilityPipeline:
    """Orchestrates observability execution in strict sequential order.

    Execution order:
    1. TelemetryCollector (produces ApiTelemetryEvent)
    2. TracingProvider (produces RequestTrace)
    3. MetricsAggregator (records event, computes metrics)
    4. SnapshotGenerator (produces ApiOperationalSnapshot)

    Invariant:
    - Observability is observational-only and side-effect free.
    - Exceptions inside the observability pipeline are isolated and logged.
    - Observability execution NEVER raises exceptions back to the transport/request handler.
    """

    def __init__(
        self,
        collector: TelemetryCollector,
        tracing_provider: TracingProvider,
        metrics_aggregator: MetricsAggregator,
        snapshot_generator: SnapshotGenerator,
        profile_id: str = "default_monitoring",
    ) -> None:
        self._collector = collector
        self._tracing_provider = tracing_provider
        self._metrics_aggregator = metrics_aggregator
        self._snapshot_generator = snapshot_generator
        self._profile_id = profile_id
        self._latest_trace: RequestTrace | None = None
        self._latest_snapshot: ApiOperationalSnapshot | None = None

    def execute(self, event: ObservationEvent) -> None:
        """Executes the observability pipeline for an incoming ObservationEvent.

        Guarantees exception isolation so request execution is never interrupted.
        """
        try:
            # 1. TelemetryCollector
            collection_result = self._collector.collect(event)
            if (
                not collection_result.success
                or collection_result.collected_event is None
            ):
                logger.warning(
                    f"Telemetry collection failed for event {event.event_id}: {collection_result.error_message}"
                )
                return

            telemetry_event = collection_result.collected_event

            # 2. TracingProvider
            self._latest_trace = self._tracing_provider.create_trace(event)

            # 3. MetricsAggregator
            self._metrics_aggregator.record(telemetry_event)

            # 4. SnapshotGenerator
            metrics = self._metrics_aggregator.compute_metrics()
            self._latest_snapshot = self._snapshot_generator.generate_snapshot(
                metrics=metrics,
                profile_id=self._profile_id,
            )
        except Exception as e:
            logger.error(
                f"ObservabilityPipeline isolated error for event {event.event_id}: {e}",
                exc_info=True,
            )

    def get_metrics(self) -> RequestMetrics:
        """Returns the current RequestMetrics calculation."""
        try:
            return self._metrics_aggregator.compute_metrics()
        except Exception as e:
            logger.error(f"Error computing metrics: {e}")
            return RequestMetrics()

    def get_latest_trace(self) -> RequestTrace | None:
        """Returns the latest generated RequestTrace."""
        return self._latest_trace

    def get_latest_snapshot(self) -> ApiOperationalSnapshot | None:
        """Returns the latest generated ApiOperationalSnapshot."""
        return self._latest_snapshot
