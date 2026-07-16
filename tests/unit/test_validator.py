"""Unit tests for dataset validation layer."""

from pathlib import Path
from unittest.mock import patch

import pytest

from src.core.datasets.artifact_models import ArtifactIdentity
from src.core.datasets.metadata import DatasetMetadata, DatasetSchema, DatasetSplit
from src.core.datasets.registry import registry
from src.core.datasets.validation_models import (
    FileConstraint,
    ManifestConstraint,
    ValidationFailureCode,
)
from src.core.datasets.validator import ArtifactValidator
from src.core.exceptions import RegistryError
from src.core.paths import ProjectPaths


@pytest.fixture
def mock_identity(tmp_path: Path) -> ArtifactIdentity:
    """Fixture providing a mock identity and isolating storage paths."""
    ProjectPaths.DATA_RAW = tmp_path

    identity = ArtifactIdentity(dataset_id="mock_data", version="1.0")

    metadata = DatasetMetadata(
        id=identity.dataset_id,
        version=identity.version,
        description="Mock",
        domain="mock",
        schema_metadata=DatasetSchema(),
        splits=(DatasetSplit.TRAIN,),
    )

    try:
        registry.register_dataset(metadata)
    except RegistryError:
        pass

    return identity


def test_validator_success(mock_identity: ArtifactIdentity, tmp_path: Path) -> None:
    """Test successful validation of all constraints."""
    dataset_dir = ProjectPaths.get_dataset_version_dir(
        mock_identity.dataset_id, mock_identity.version
    )
    dataset_dir.mkdir(parents=True)

    file_path = dataset_dir / "data.txt"
    file_path.write_text("content")

    manifest_path = dataset_dir / "manifest.json"
    manifest_path.write_text('{"key": "value"}')

    validator = ArtifactValidator()

    constraints = (
        FileConstraint(id="c1", target_path="data.txt", min_size_bytes=5),
        ManifestConstraint(id="c2", target_path="manifest.json", is_jsonl=False),
    )

    report = validator.validate(mock_identity, constraints)

    assert report.is_valid is True
    assert len(report.results) == 2
    assert all(r.passed for r in report.results)


def test_validator_missing_file(
    mock_identity: ArtifactIdentity, tmp_path: Path
) -> None:
    """Test validation failure for missing file."""
    dataset_dir = ProjectPaths.get_dataset_version_dir(
        mock_identity.dataset_id, mock_identity.version
    )
    dataset_dir.mkdir(parents=True)

    validator = ArtifactValidator()

    constraints = (FileConstraint(id="c1", target_path="missing.txt"),)

    report = validator.validate(mock_identity, constraints)

    assert report.is_valid is False
    assert report.results[0].passed is False
    assert report.results[0].failure_code == ValidationFailureCode.MISSING_FILE


def test_validator_insufficient_size(
    mock_identity: ArtifactIdentity, tmp_path: Path
) -> None:
    """Test validation failure for a file that is too small."""
    dataset_dir = ProjectPaths.get_dataset_version_dir(
        mock_identity.dataset_id, mock_identity.version
    )
    dataset_dir.mkdir(parents=True)

    file_path = dataset_dir / "data.txt"
    file_path.write_text("tiny")

    validator = ArtifactValidator()

    constraints = (FileConstraint(id="c1", target_path="data.txt", min_size_bytes=100),)

    report = validator.validate(mock_identity, constraints)

    assert report.is_valid is False
    assert report.results[0].failure_code == ValidationFailureCode.INSUFFICIENT_SIZE


def test_validator_unreadable_file(
    mock_identity: ArtifactIdentity, tmp_path: Path
) -> None:
    """Test validation failure for an unreadable file."""
    dataset_dir = ProjectPaths.get_dataset_version_dir(
        mock_identity.dataset_id, mock_identity.version
    )
    dataset_dir.mkdir(parents=True)

    file_path = dataset_dir / "data.txt"
    file_path.write_text("content")

    validator = ArtifactValidator()

    constraints = (FileConstraint(id="c1", target_path="data.txt"),)

    # Mock os.access to return False for read permissions
    with patch("os.access", return_value=False):
        report = validator.validate(mock_identity, constraints)

    assert report.is_valid is False
    assert report.results[0].failure_code == ValidationFailureCode.UNREADABLE


def test_validator_corrupt_manifest(
    mock_identity: ArtifactIdentity, tmp_path: Path
) -> None:
    """Test validation failure for malformed JSON manifest."""
    dataset_dir = ProjectPaths.get_dataset_version_dir(
        mock_identity.dataset_id, mock_identity.version
    )
    dataset_dir.mkdir(parents=True)

    manifest_path = dataset_dir / "manifest.json"
    manifest_path.write_text("{invalid_json")

    validator = ArtifactValidator()

    constraints = (
        ManifestConstraint(id="c1", target_path="manifest.json", is_jsonl=False),
    )

    report = validator.validate(mock_identity, constraints)

    assert report.is_valid is False
    assert report.results[0].failure_code == ValidationFailureCode.MANIFEST_CORRUPT


def test_validator_jsonl_manifest(
    mock_identity: ArtifactIdentity, tmp_path: Path
) -> None:
    """Test successful validation of JSONL manifest."""
    dataset_dir = ProjectPaths.get_dataset_version_dir(
        mock_identity.dataset_id, mock_identity.version
    )
    dataset_dir.mkdir(parents=True)

    manifest_path = dataset_dir / "manifest.jsonl"
    manifest_path.write_text('{"id": 1}\n{"id": 2}\n')

    validator = ArtifactValidator()

    constraints = (
        ManifestConstraint(id="c1", target_path="manifest.jsonl", is_jsonl=True),
    )

    report = validator.validate(mock_identity, constraints)

    assert report.is_valid is True
