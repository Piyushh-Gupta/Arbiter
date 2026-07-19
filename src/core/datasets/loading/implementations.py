"""Concrete implementations of the loading framework strategies."""

import json

from pydantic import ValidationError

from src.core.datasets.export_models import SerializedArtifact
from src.core.datasets.loading.base import BaseLoader
from src.core.datasets.loading_models import (
    ArbiterDataset,
    JsonlLoadingDefinition,
    LoadingDefinition,
)
from src.core.datasets.mapping_models import TaskRecord
from src.core.datasets.serialization_models import DatasetManifest
from src.core.exceptions import LoadingConfigurationError, LoadingExecutionError


class JsonlLoader(BaseLoader):
    """Stateless execution strategy reconstructing ArbiterDataset from JSONL artifacts."""

    def validate_compatibility(self, definition: LoadingDefinition) -> None:
        """Fails fast if the definition is not a JsonlLoadingDefinition."""
        if not isinstance(definition, JsonlLoadingDefinition):
            raise LoadingConfigurationError(
                f"JsonlLoader requires JsonlLoadingDefinition, got {type(definition).__name__}"
            )

    def load(
        self, artifact: SerializedArtifact, definition: LoadingDefinition
    ) -> ArbiterDataset:
        """
        Executes the JSONL reconstruction process.
        """
        # Strict validation
        if not isinstance(definition, JsonlLoadingDefinition):
            raise LoadingConfigurationError(
                f"JsonlLoader requires JsonlLoadingDefinition, got {type(definition).__name__}"
            )

        # Artifact Discovery - Hard-coded established filenames
        manifest_path = artifact.root_path / "manifest.json"
        dataset_path = artifact.root_path / "dataset.jsonl"

        # Manifest Loading
        try:
            manifest_json = manifest_path.read_text(encoding="utf-8")
            manifest_dict = json.loads(manifest_json)
            manifest = DatasetManifest.model_validate(manifest_dict)
        except (OSError, json.JSONDecodeError, ValidationError) as e:
            raise LoadingExecutionError(
                f"Failed to load or parse manifest.json: {e}"
            ) from e

        # Dataset Loading
        records = []
        try:
            with dataset_path.open(mode="r", encoding=definition.encoding) as f:
                for line in f:
                    if not line.strip():
                        continue

                    try:
                        record_dict = json.loads(line)
                    except json.JSONDecodeError as e:
                        raise LoadingExecutionError(
                            f"Malformed JSON on line: {line}"
                        ) from e

                    try:
                        # Attempt to dynamically resolve the exact TaskRecord subclass
                        record = None

                        # Sort subclasses to prioritize those with more fields to avoid premature fallback to base
                        subclasses = sorted(
                            TaskRecord.__subclasses__(),
                            key=lambda c: len(c.model_fields),
                            reverse=True,
                        )

                        for cls in subclasses:
                            # Verify if the dictionary contains all required fields of the subclass
                            required_fields = {
                                k
                                for k, v in cls.model_fields.items()
                                if v.is_required()
                            }
                            if required_fields.issubset(record_dict.keys()):
                                try:
                                    record = cls.model_validate(record_dict)
                                    break
                                except ValidationError:
                                    continue

                        if record is None:
                            record = TaskRecord.model_validate(record_dict)

                        records.append(record)
                    except ValidationError as e:
                        raise LoadingExecutionError(
                            f"Schema validation failed: {e}"
                        ) from e

        except OSError as e:
            raise LoadingExecutionError(f"Failed to read dataset.jsonl: {e}") from e

        return ArbiterDataset(manifest=manifest, records=tuple(records))
