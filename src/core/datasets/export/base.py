"""Protocol defining the interface for Dataset Export strategies."""

from typing import Protocol

from src.core.datasets.export_models import ExportDefinition, SerializedArtifact


class BaseExporter(Protocol):
    """Protocol for concrete dataset export strategies."""

    def export(
        self, artifact: SerializedArtifact, definition: ExportDefinition
    ) -> None:
        """Executes the export side-effect for the given serialized artifact."""
        ...

    def validate_compatibility(self, definition: ExportDefinition) -> None:
        """Validates that the provided definition matches the exporter's expectations."""
        ...
