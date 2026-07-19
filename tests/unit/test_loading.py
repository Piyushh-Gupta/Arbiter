"""Unit tests for the M7.1 Dataset Loading Framework."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.core.datasets.export_models import SerializedArtifact
from src.core.datasets.loader import DatasetLoader
from src.core.datasets.loading.base import BaseLoader
from src.core.datasets.loading_models import ArbiterDataset, LoadingDefinition
from src.core.datasets.mapping_models import FactVerificationRecord, TaskRecord
from src.core.datasets.normalization_models import ProvenanceMetadata
from src.core.datasets.serialization_models import DatasetManifest, SerializationFormat
from src.core.exceptions import LoadingConfigurationError, LoadingExecutionError


@pytest.fixture
def dummy_manifest() -> DatasetManifest:
    return DatasetManifest(
        manifest_version="1.0",
        serialization_format=SerializationFormat.JSONL,
        dataset_id="test",
        dataset_version="1.0",
        record_count=1,
    )


@pytest.fixture
def dummy_records() -> tuple[TaskRecord, ...]:
    return (
        FactVerificationRecord(
            provenance=ProvenanceMetadata(record_index=0),
            claim="test claim",
            evidence="test evidence",
            label="SUPPORTS",
        ),
    )


@pytest.fixture
def dummy_dataset(
    dummy_manifest: DatasetManifest, dummy_records: tuple[TaskRecord, ...]
) -> ArbiterDataset:
    return ArbiterDataset(manifest=dummy_manifest, records=dummy_records)


@pytest.fixture
def dummy_artifact(tmp_path: Path) -> SerializedArtifact:
    return SerializedArtifact(root_path=tmp_path)


@pytest.fixture
def dummy_definition() -> LoadingDefinition:
    return LoadingDefinition()


def test_loading_definition_immutability() -> None:
    """Test that LoadingDefinition is strictly immutable."""
    definition = LoadingDefinition()
    with pytest.raises(Exception):
        # Using pytest.raises(Exception) to catch Pydantic's ValidationError/FrozenError
        definition.some_attr = "mutated"  # type: ignore[attr-defined]


def test_arbiter_dataset_immutability(
    dummy_manifest: DatasetManifest, dummy_records: tuple[TaskRecord, ...]
) -> None:
    """Test that ArbiterDataset is strictly immutable."""
    dataset = ArbiterDataset(manifest=dummy_manifest, records=dummy_records)
    with pytest.raises(Exception):
        dataset.records = ()


def test_dataset_loader_orchestration_success(
    dummy_artifact: SerializedArtifact,
    dummy_definition: LoadingDefinition,
    dummy_dataset: ArbiterDataset,
) -> None:
    """Test DatasetLoader purely orchestrates execution preserving object identities."""
    loader = DatasetLoader()
    mock_strategy = MagicMock(spec=BaseLoader)
    mock_strategy.load.return_value = dummy_dataset

    result = loader.load(dummy_artifact, dummy_definition, mock_strategy)

    # Verify compatibility validation was executed exactly once
    mock_strategy.validate_compatibility.assert_called_once_with(dummy_definition)

    # Verify load was delegated exactly once preserving artifact and definition identities
    mock_strategy.load.assert_called_once_with(dummy_artifact, dummy_definition)

    # Verify return object identity is perfectly preserved
    assert result is dummy_dataset


def test_dataset_loader_compatibility_fail_fast(
    dummy_artifact: SerializedArtifact,
    dummy_definition: LoadingDefinition,
) -> None:
    """Test that incompatibility natively halts execution before loading begins."""
    loader = DatasetLoader()
    mock_strategy = MagicMock(spec=BaseLoader)
    mock_strategy.validate_compatibility.side_effect = LoadingConfigurationError(
        "Incompatible"
    )

    with pytest.raises(LoadingConfigurationError, match="Incompatible"):
        loader.load(dummy_artifact, dummy_definition, mock_strategy)

    # Load should never be invoked if compatibility validation fails
    mock_strategy.load.assert_not_called()


def test_dataset_loader_execution_fail_fast(
    dummy_artifact: SerializedArtifact,
    dummy_definition: LoadingDefinition,
) -> None:
    """Test that execution failures propagate seamlessly as LoadingExecutionError."""
    loader = DatasetLoader()
    mock_strategy = MagicMock(spec=BaseLoader)
    mock_strategy.load.side_effect = LoadingExecutionError("IO Failure")

    with pytest.raises(LoadingExecutionError, match="IO Failure"):
        loader.load(dummy_artifact, dummy_definition, mock_strategy)

    mock_strategy.validate_compatibility.assert_called_once_with(dummy_definition)
    mock_strategy.load.assert_called_once_with(dummy_artifact, dummy_definition)


def test_dataset_loader_stateless_execution(
    dummy_artifact: SerializedArtifact,
    dummy_definition: LoadingDefinition,
    dummy_dataset: ArbiterDataset,
) -> None:
    """Test multiple orchestrations maintain no internal state."""
    loader = DatasetLoader()
    mock_strategy = MagicMock(spec=BaseLoader)
    mock_strategy.load.return_value = dummy_dataset

    # Execute multiple times
    for _ in range(5):
        result = loader.load(dummy_artifact, dummy_definition, mock_strategy)
        assert result is dummy_dataset

    assert mock_strategy.validate_compatibility.call_count == 5
    assert mock_strategy.load.call_count == 5
