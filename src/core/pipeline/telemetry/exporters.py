"""Implementations of telemetry exporters."""

import logging
import os
from datetime import datetime, timezone

from src.core.exceptions import PipelineStageExecutionError, TelemetryConfigurationError
from src.core.pipeline.telemetry.base import BaseTelemetryExporter
from src.core.pipeline.telemetry.profile_models import (
    JsonTelemetryExporterDefinition,
    LogTelemetryExporterDefinition,
    TelemetryExporterDefinition,
)
from src.core.pipeline.telemetry.telemetry_models import (
    PipelineTelemetryReport,
    PipelineTelemetrySnapshot,
)

logger = logging.getLogger("arbiter.telemetry")


class LogTelemetryExporter(BaseTelemetryExporter):
    """Telemetry exporter logging structured summaries."""

    def __init__(self) -> None:
        self._definition: LogTelemetryExporterDefinition | None = None

    def validate_compatibility(self, definition: TelemetryExporterDefinition) -> None:
        """Validates configuration compatibility."""
        if not isinstance(definition, LogTelemetryExporterDefinition):
            raise TelemetryConfigurationError(
                f"Incompatible definition type for LogTelemetryExporter: {type(definition)}"
            )
        self._definition = definition

    def export(self, snapshot: PipelineTelemetrySnapshot) -> PipelineTelemetryReport:
        """Logs the snapshot at the configured logging level."""
        level_str = self._definition.log_level if self._definition else "INFO"
        include_stages = (
            self._definition.include_stage_breakdown if self._definition else True
        )

        level = getattr(logging, level_str.upper(), logging.INFO)

        breakdown_parts = []
        if include_stages:
            for stage in snapshot.stage_aggregations:
                breakdown_parts.append(
                    f"{stage.stage_id}({stage.profile_id}): count={stage.execution_count}, "
                    f"success_rate={stage.success_rate:.2%}, mean_latency={stage.mean_latency_ms:.2f}ms"
                )

        breakdown_str = " | " + ", ".join(breakdown_parts) if breakdown_parts else ""

        content = (
            f"Pipeline={snapshot.pipeline_id} | total_executions={snapshot.total_executions} | "
            f"overall_success_rate={snapshot.overall_success_rate:.2%} | "
            f"mean_total_latency={snapshot.mean_total_latency_ms:.2f}ms{breakdown_str}"
        )

        logger.log(level, content)

        return PipelineTelemetryReport(
            snapshot=snapshot,
            format="log",
            content=content,
            generated_at=datetime.now(timezone.utc),
        )


class JsonTelemetryExporter(BaseTelemetryExporter):
    """Telemetry exporter writing JSON snapshots to a file."""

    def __init__(self) -> None:
        self._definition: JsonTelemetryExporterDefinition | None = None

    def validate_compatibility(self, definition: TelemetryExporterDefinition) -> None:
        """Validates configuration compatibility."""
        if not isinstance(definition, JsonTelemetryExporterDefinition):
            raise TelemetryConfigurationError(
                f"Incompatible definition type for JsonTelemetryExporter: {type(definition)}"
            )
        self._definition = definition

    def export(self, snapshot: PipelineTelemetrySnapshot) -> PipelineTelemetryReport:
        """Writes the JSON representation of the snapshot to the output path."""
        if not self._definition:
            raise TelemetryConfigurationError(
                "Exporter not initialized with definition."
            )

        indent = 2 if self._definition.pretty_print else None
        content = snapshot.model_dump_json(indent=indent)

        try:
            output_path = self._definition.output_path
            dir_name = os.path.dirname(output_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            raise PipelineStageExecutionError(
                f"Failed to write telemetry JSON snapshot to {self._definition.output_path}: {e}"
            ) from e

        return PipelineTelemetryReport(
            snapshot=snapshot,
            format="json",
            content=content,
            generated_at=datetime.now(timezone.utc),
        )
