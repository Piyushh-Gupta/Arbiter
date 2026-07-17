"""Concrete implementations for the Dataset Export subsystem."""

import shutil

from src.core.datasets.export.base import BaseExporter
from src.core.datasets.export_models import (
    ExportDefinition,
    LocalExportDefinition,
    SerializedArtifact,
)
from src.core.exceptions import ExportConfigurationError, ExportExecutionError


class LocalExporter(BaseExporter):
    """Exports serialized artifacts to the local filesystem."""

    def export(
        self, artifact: SerializedArtifact, definition: ExportDefinition
    ) -> None:
        """Executes the export side-effect securely on the local filesystem."""
        assert isinstance(definition, LocalExportDefinition)

        try:
            # Ensure parent exists
            definition.destination_root.parent.mkdir(parents=True, exist_ok=True)

            # Handle overwrite
            if definition.destination_root.exists():
                if not definition.overwrite_existing:
                    raise ExportExecutionError(
                        f"Destination {definition.destination_root} already exists and overwrite is disabled."
                    )
                # Overwrite enabled: explicitly remove entirely
                if definition.destination_root.is_dir():
                    shutil.rmtree(definition.destination_root)
                else:
                    definition.destination_root.unlink()

            # Perform the exact copy
            if artifact.root_path.is_dir():
                shutil.copytree(
                    artifact.root_path,
                    definition.destination_root,
                    dirs_exist_ok=False,
                )
            else:
                shutil.copy2(artifact.root_path, definition.destination_root)
        except ExportExecutionError:
            raise
        except Exception as e:
            raise ExportExecutionError(
                f"Failed to export to {definition.destination_root}: {e}"
            ) from e

    def validate_compatibility(self, definition: ExportDefinition) -> None:
        """Ensures the definition is a LocalExportDefinition."""
        if not isinstance(definition, LocalExportDefinition):
            raise ExportConfigurationError(
                f"LocalExporter requires LocalExportDefinition, got {type(definition).__name__}"
            )
