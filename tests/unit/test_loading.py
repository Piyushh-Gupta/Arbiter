"""Unit tests for the M7.1 Dataset Loading Framework."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.core.datasets.export_models import SerializedArtifact
from src.core.datasets.loader import DatasetLoader
from src.core.datasets.loading.base import BaseLoader
from src.core.datasets.loading.implementations import JsonlLoader
from src.core.datasets.loading_models import (
    ArbiterDataset,
    JsonlLoadingDefinition,
    LoadingDefinition,
)
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


@pytest.fixture
def jsonl_loader() -> JsonlLoader:
    return JsonlLoader()


@pytest.fixture
def jsonl_definition() -> JsonlLoadingDefinition:
    return JsonlLoadingDefinition(encoding="utf-8")


def _setup_jsonl_artifact(
    tmp_path: Path, manifest: DatasetManifest, records: list[dict[str, Any]]
) -> SerializedArtifact:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")

    dataset_path = tmp_path / "dataset.jsonl"
    with dataset_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    return SerializedArtifact(root_path=tmp_path)


def test_jsonl_loader_compatibility(
    jsonl_loader: JsonlLoader, jsonl_definition: JsonlLoadingDefinition
) -> None:
    """Test compatibility validation."""
    jsonl_loader.validate_compatibility(jsonl_definition)
    with pytest.raises(LoadingConfigurationError):
        jsonl_loader.validate_compatibility(LoadingDefinition())


def test_jsonl_loader_empty_dataset(
    jsonl_loader: JsonlLoader,
    jsonl_definition: JsonlLoadingDefinition,
    dummy_manifest: DatasetManifest,
    tmp_path: Path,
) -> None:
    """Test loading a JSONL dataset with zero records."""
    artifact = _setup_jsonl_artifact(tmp_path, dummy_manifest, [])
    dataset = jsonl_loader.load(artifact, jsonl_definition)

    assert dataset.manifest == dummy_manifest
    assert len(dataset.records) == 0


def test_jsonl_loader_single_record(
    jsonl_loader: JsonlLoader,
    jsonl_definition: JsonlLoadingDefinition,
    dummy_manifest: DatasetManifest,
    tmp_path: Path,
) -> None:
    """Test loading a JSONL dataset with one record."""
    record_dict = {
        "provenance": {"record_index": 0},
        "claim": "Test claim",
        "evidence": "Test evidence",
        "label": "SUPPORTS",
    }
    artifact = _setup_jsonl_artifact(tmp_path, dummy_manifest, [record_dict])
    dataset = jsonl_loader.load(artifact, jsonl_definition)

    assert dataset.manifest == dummy_manifest
    assert len(dataset.records) == 1
    assert isinstance(dataset.records[0], FactVerificationRecord)
    assert dataset.records[0].claim == "Test claim"


def test_jsonl_loader_multiple_records(
    jsonl_loader: JsonlLoader,
    jsonl_definition: JsonlLoadingDefinition,
    dummy_manifest: DatasetManifest,
    tmp_path: Path,
) -> None:
    """Test loading a JSONL dataset with multiple records."""
    records = [
        {
            "provenance": {"record_index": i},
            "claim": f"Claim {i}",
            "evidence": None,
            "label": None,
        }
        for i in range(3)
    ]
    artifact = _setup_jsonl_artifact(tmp_path, dummy_manifest, records)
    dataset = jsonl_loader.load(artifact, jsonl_definition)

    assert len(dataset.records) == 3
    for i, record in enumerate(dataset.records):
        assert isinstance(record, FactVerificationRecord)
        assert record.claim == f"Claim {i}"


def test_jsonl_loader_missing_artifact(
    jsonl_loader: JsonlLoader,
    jsonl_definition: JsonlLoadingDefinition,
    tmp_path: Path,
) -> None:
    """Test fail-fast when files are missing."""
    artifact = SerializedArtifact(root_path=tmp_path)
    with pytest.raises(
        LoadingExecutionError, match="Failed to load or parse manifest.json"
    ):
        jsonl_loader.load(artifact, jsonl_definition)


def test_jsonl_loader_missing_jsonl_file(
    jsonl_loader: JsonlLoader,
    jsonl_definition: JsonlLoadingDefinition,
    dummy_manifest: DatasetManifest,
    tmp_path: Path,
) -> None:
    """Test fail-fast when JSONL is missing but manifest exists."""
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(dummy_manifest.model_dump_json(), encoding="utf-8")
    artifact = SerializedArtifact(root_path=tmp_path)

    with pytest.raises(LoadingExecutionError, match="Failed to read dataset.jsonl"):
        jsonl_loader.load(artifact, jsonl_definition)


def test_jsonl_loader_malformed_json(
    jsonl_loader: JsonlLoader,
    jsonl_definition: JsonlLoadingDefinition,
    dummy_manifest: DatasetManifest,
    tmp_path: Path,
) -> None:
    """Test fail-fast on corrupted JSON lines."""
    artifact = _setup_jsonl_artifact(tmp_path, dummy_manifest, [])
    with (tmp_path / "dataset.jsonl").open("w") as f:
        f.write("{invalid json\n")

    with pytest.raises(LoadingExecutionError, match="Malformed JSON on line"):
        jsonl_loader.load(artifact, jsonl_definition)


def test_jsonl_loader_invalid_schema(
    jsonl_loader: JsonlLoader,
    jsonl_definition: JsonlLoadingDefinition,
    dummy_manifest: DatasetManifest,
    tmp_path: Path,
) -> None:
    """Test fail-fast on JSON missing required schema fields."""
    record_dict = {"invalid": "schema"}
    artifact = _setup_jsonl_artifact(tmp_path, dummy_manifest, [record_dict])

    with pytest.raises(LoadingExecutionError, match="Schema validation failed"):
        jsonl_loader.load(artifact, jsonl_definition)


def test_jsonl_loader_deterministic_reconstruction(
    jsonl_loader: JsonlLoader,
    jsonl_definition: JsonlLoadingDefinition,
    dummy_manifest: DatasetManifest,
    tmp_path: Path,
) -> None:
    """Test stateless execution produces identical outcomes repeatedly."""
    records = [
        {
            "provenance": {"record_index": 0},
            "claim": "C1",
            "evidence": None,
            "label": None,
        },
        {
            "provenance": {"record_index": 1},
            "claim": "C2",
            "evidence": None,
            "label": None,
        },
    ]
    artifact = _setup_jsonl_artifact(tmp_path, dummy_manifest, records)

    dataset_a = jsonl_loader.load(artifact, jsonl_definition)
    dataset_b = jsonl_loader.load(artifact, jsonl_definition)

    # Validate deterministic object values (not identity, but equality)
    assert dataset_a.manifest == dataset_b.manifest
    assert len(dataset_a.records) == len(dataset_b.records)
    for a, b in zip(dataset_a.records, dataset_b.records):
        assert a == b
