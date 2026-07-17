"""Concrete implementations for the Dataset Export subsystem."""

import shutil

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError  # type: ignore[attr-defined]

from src.core.datasets.export.base import BaseExporter
from src.core.datasets.export_models import (
    ExportDefinition,
    HuggingFaceExportDefinition,
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


class HuggingFaceExporter(BaseExporter):
    """Exports serialized artifacts to the Hugging Face Hub."""

    def export(
        self, artifact: SerializedArtifact, definition: ExportDefinition
    ) -> None:
        """Executes the export side-effect securely to Hugging Face."""
        assert isinstance(definition, HuggingFaceExportDefinition)

        try:
            api = HfApi()

            if definition.create_repo_if_missing:
                api.create_repo(
                    repo_id=definition.repository_id,
                    repo_type=definition.repository_type.value,
                    private=definition.private,
                    exist_ok=True,
                )
            else:
                # Expect repository to exist; raising ExportExecutionError if not
                try:
                    api.dataset_info(repo_id=definition.repository_id)
                except Exception as e:
                    raise ExportExecutionError(
                        f"Repository {definition.repository_id} does not exist and create_repo_if_missing=False."
                    ) from e

            if artifact.root_path.is_dir():
                api.upload_folder(
                    folder_path=str(artifact.root_path),
                    repo_id=definition.repository_id,
                    repo_type=definition.repository_type.value,
                    revision=definition.revision,
                    commit_message=definition.commit_message,
                )
            else:
                api.upload_file(
                    path_or_fileobj=str(artifact.root_path),
                    path_in_repo=artifact.root_path.name,
                    repo_id=definition.repository_id,
                    repo_type=definition.repository_type.value,
                    revision=definition.revision,
                    commit_message=definition.commit_message,
                )

        except ExportExecutionError:
            raise
        except HfHubHTTPError as e:
            raise ExportExecutionError(f"Hugging Face HTTP error: {e}") from e
        except Exception as e:
            raise ExportExecutionError(f"Hugging Face export failed: {e}") from e

    def validate_compatibility(self, definition: ExportDefinition) -> None:
        """Ensures the definition is a HuggingFaceExportDefinition."""
        if not isinstance(definition, HuggingFaceExportDefinition):
            raise ExportConfigurationError(
                f"HuggingFaceExporter requires HuggingFaceExportDefinition, got {type(definition).__name__}"
            )
