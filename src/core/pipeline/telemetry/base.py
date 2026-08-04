"""Stateless protocols for the telemetry subsystem."""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from src.core.pipeline.pipeline_models import PipelineExecutionResult
    from src.core.pipeline.telemetry.profile_models import TelemetryExporterDefinition

from src.core.pipeline.telemetry.telemetry_models import (
    PipelineTelemetryEvent,
    PipelineTelemetryReport,
    PipelineTelemetrySnapshot,
)


@runtime_checkable
class BaseTelemetryCollector(Protocol):
    """Stateless protocol for telemetry accumulation."""

    def record(self, event: PipelineTelemetryEvent) -> None:
        """Records a single telemetry event."""
        ...

    def snapshot(self) -> PipelineTelemetrySnapshot:
        """Produces an immutable aggregated snapshot of all recorded events."""
        ...

    def reset(self) -> None:
        """Clears all accumulated telemetry data."""
        ...


@runtime_checkable
class BaseTelemetryExporter(Protocol):
    """Stateless protocol for telemetry export."""

    def validate_compatibility(self, definition: "TelemetryExporterDefinition") -> None:
        """Validates configuration compatibility.

        Raises:
            TelemetryConfigurationError: If definition is invalid/incompatible.
        """
        ...

    def export(self, snapshot: PipelineTelemetrySnapshot) -> PipelineTelemetryReport:
        """Exports the snapshot and returns an immutable report."""
        ...


@runtime_checkable
class BaseTelemetryEventFactory(Protocol):
    """Stateless protocol for constructing PipelineTelemetryEvent from PipelineExecutionResult."""

    def from_result(self, result: "PipelineExecutionResult") -> PipelineTelemetryEvent:
        """Constructs a PipelineTelemetryEvent from an immutable PipelineExecutionResult."""
        ...
