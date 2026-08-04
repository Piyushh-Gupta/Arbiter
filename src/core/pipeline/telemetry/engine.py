"""Telemetry engine implementation coordinating observation and export."""

import logging

from src.core.pipeline.pipeline_models import PipelineExecutionResult
from src.core.pipeline.telemetry.base import (
    BaseTelemetryCollector,
    BaseTelemetryEventFactory,
)
from src.core.pipeline.telemetry.profile_models import (
    PipelineTelemetryDefinition,
    TelemetryExporterRegistry,
)
from src.core.pipeline.telemetry.telemetry_models import PipelineTelemetryReport

logger = logging.getLogger("arbiter.telemetry")


class PipelineTelemetryEngine:
    """Coordinator that routes pipeline execution telemetry to collectors and exporters."""

    def __init__(
        self,
        definition: PipelineTelemetryDefinition,
        collector: BaseTelemetryCollector,
        exporter_registry: TelemetryExporterRegistry,
        event_factory: BaseTelemetryEventFactory,
    ) -> None:
        self._definition = definition
        self._collector = collector
        self._exporter_registry = exporter_registry
        self._event_factory = event_factory

    def observe(self, result: PipelineExecutionResult) -> None:
        """Converts result to event and records it. Optionally triggers exporters."""
        if not self._definition.enabled:
            return

        try:
            event = self._event_factory.from_result(result)
            self._collector.record(event)
        except Exception as e:
            logger.error(f"Telemetry observation failed: {e}", exc_info=True)
            return

        if self._definition.snapshot_on_every_execution:
            self.export_snapshot()

    def export_snapshot(self) -> tuple[PipelineTelemetryReport, ...]:
        """Generates a snapshot and exports it via all configured active exporters."""
        if not self._definition.enabled:
            return ()

        try:
            snapshot = self._collector.snapshot()
        except Exception as e:
            logger.error(f"Failed to generate telemetry snapshot: {e}", exc_info=True)
            return ()

        reports = []
        for profile_id in self._definition.active_exporter_profile_ids:
            try:
                profile = self._exporter_registry.resolve(profile_id)
                if profile.definition.enabled:
                    report = profile.exporter.export(snapshot)
                    reports.append(report)
            except Exception as e:
                logger.error(
                    f"Telemetry export failed for profile '{profile_id}': {e}",
                    exc_info=True,
                )

        return tuple(reports)

    def reset(self) -> None:
        """Resets the internal collector state."""
        self._collector.reset()
