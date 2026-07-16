"""Unit tests for dataset manifest models and loader."""

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from src.core.constants import MANIFEST_FILENAME
from src.core.datasets.artifact_models import ArtifactIdentity
from src.core.datasets.manifest_loader import ManifestLoader
from src.core.datasets.manifest_models import (
    DatasetManifest,
    ManifestSchemaVersion,
    SupportedTask,
)
from src.core.exceptions import (
    ManifestNotFoundError,
    ManifestParseError,
    UnsupportedSchemaVersionError,
)
from src.core.paths import ProjectPaths


@pytest.fixture
def mock_identity() -> ArtifactIdentity:
    return ArtifactIdentity(dataset_id="test_dataset", version="1.0.0")


@pytest.fixture
def valid_manifest_data() -> dict[str, Any]:
    return {
        "dataset_name": "Test Dataset",
        "identity": {"dataset_id": "test_dataset", "version": "1.0.0"},
        "description": "A test dataset",
        "source": "example.com",
        "license": "MIT",
        "citation": "Test et al. 2026",
        "supported_tasks": ["qa", "classification"],
        "languages": ["en"],
        "domain": "general",
        "dataset_structure_summary": "Simple JSONL with text and label",
        "expected_splits": ["train", "test"],
        "artifact_inventory": [
            {
                "filename": "train.jsonl",
                "file_type": "jsonl",
                "role": "train_data",
                "description": "Training split",
            }
        ],
        "creation_timestamp": "2026-07-16T12:00:00Z",
        "schema_version": {"version": "1.0.0"},
    }


def test_manifest_schema_version_validation() -> None:
    """Test validation of ManifestSchemaVersion."""
    valid = ManifestSchemaVersion(version="1.0.0")
    assert valid.major == 1
    assert valid.minor == 0
    assert valid.patch == 0

    with pytest.raises(UnsupportedSchemaVersionError):
        ManifestSchemaVersion(version="1.0")

    with pytest.raises(UnsupportedSchemaVersionError):
        ManifestSchemaVersion(version="v1.0.0")


def test_manifest_immutability(valid_manifest_data: dict[str, Any]) -> None:
    """Test that DatasetManifest is immutable."""
    manifest = DatasetManifest.model_validate(valid_manifest_data)

    with pytest.raises(ValidationError):
        manifest.description = "New description"


def test_manifest_loader_success(
    mock_identity: ArtifactIdentity, valid_manifest_data: dict[str, Any], tmp_path: Path
) -> None:
    """Test successful manifest loading."""
    ProjectPaths.DATA_RAW = tmp_path
    version_dir = ProjectPaths.get_dataset_version_dir(
        mock_identity.dataset_id, mock_identity.version
    )
    version_dir.mkdir(parents=True)

    manifest_path = version_dir / MANIFEST_FILENAME
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(valid_manifest_data, f)

    manifest = ManifestLoader.load_manifest(mock_identity)

    assert manifest.dataset_name == "Test Dataset"
    assert manifest.identity.dataset_id == "test_dataset"
    assert manifest.schema_version.version == "1.0.0"
    assert len(manifest.supported_tasks) == 2
    assert SupportedTask.QA in manifest.supported_tasks


def test_manifest_loader_not_found(
    mock_identity: ArtifactIdentity, tmp_path: Path
) -> None:
    """Test loading fails when manifest does not exist."""
    ProjectPaths.DATA_RAW = tmp_path

    with pytest.raises(ManifestNotFoundError):
        ManifestLoader.load_manifest(mock_identity)


def test_manifest_loader_malformed_json(
    mock_identity: ArtifactIdentity, tmp_path: Path
) -> None:
    """Test loading fails on invalid JSON."""
    ProjectPaths.DATA_RAW = tmp_path
    version_dir = ProjectPaths.get_dataset_version_dir(
        mock_identity.dataset_id, mock_identity.version
    )
    version_dir.mkdir(parents=True)

    manifest_path = version_dir / MANIFEST_FILENAME
    manifest_path.write_text("{malformed_json")

    with pytest.raises(ManifestParseError, match="Malformed JSON"):
        ManifestLoader.load_manifest(mock_identity)


def test_manifest_loader_schema_validation_error(
    mock_identity: ArtifactIdentity, valid_manifest_data: dict[str, Any], tmp_path: Path
) -> None:
    """Test loading fails when required fields are missing."""
    ProjectPaths.DATA_RAW = tmp_path
    version_dir = ProjectPaths.get_dataset_version_dir(
        mock_identity.dataset_id, mock_identity.version
    )
    version_dir.mkdir(parents=True)

    # Remove required field
    del valid_manifest_data["description"]

    manifest_path = version_dir / MANIFEST_FILENAME
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(valid_manifest_data, f)

    with pytest.raises(ManifestParseError, match="Schema validation failed"):
        ManifestLoader.load_manifest(mock_identity)


def test_manifest_loader_unsupported_version(
    mock_identity: ArtifactIdentity, valid_manifest_data: dict[str, Any], tmp_path: Path
) -> None:
    """Test loading fails on unsupported schema version."""
    ProjectPaths.DATA_RAW = tmp_path
    version_dir = ProjectPaths.get_dataset_version_dir(
        mock_identity.dataset_id, mock_identity.version
    )
    version_dir.mkdir(parents=True)

    valid_manifest_data["schema_version"] = {"version": "2.0.0"}

    manifest_path = version_dir / MANIFEST_FILENAME
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(valid_manifest_data, f)

    with pytest.raises(
        UnsupportedSchemaVersionError, match="Unsupported manifest schema version"
    ):
        ManifestLoader.load_manifest(mock_identity)
