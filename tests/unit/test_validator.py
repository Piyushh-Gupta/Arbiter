"""Unit tests for dataset validation layer."""

from pathlib import Path
from unittest.mock import patch

from src.core.datasets.validation_models import (
    FileConstraint,
    ManifestConstraint,
    ValidationFailureCode,
)
from src.core.datasets.validator import DatasetValidator


def test_validator_success(tmp_path: Path) -> None:
    """Test successful validation of all constraints."""
    dataset_dir = tmp_path / "mock_data"
    dataset_dir.mkdir()

    file_path = dataset_dir / "data.txt"
    file_path.write_text("content")

    manifest_path = dataset_dir / "manifest.json"
    manifest_path.write_text('{"key": "value"}')

    validator = DatasetValidator()
    validator.raw_dir = tmp_path

    constraints = (
        FileConstraint(id="c1", target_path="data.txt", min_size_bytes=5),
        ManifestConstraint(id="c2", target_path="manifest.json", is_jsonl=False),
    )

    report = validator.validate("mock_data", "1.0", constraints)

    assert report.is_valid is True
    assert len(report.results) == 2
    assert all(r.passed for r in report.results)


def test_validator_missing_file(tmp_path: Path) -> None:
    """Test validation failure for missing file."""
    dataset_dir = tmp_path / "mock_data"
    dataset_dir.mkdir()

    validator = DatasetValidator()
    validator.raw_dir = tmp_path

    constraints = (FileConstraint(id="c1", target_path="missing.txt"),)

    report = validator.validate("mock_data", "1.0", constraints)

    assert report.is_valid is False
    assert report.results[0].passed is False
    assert report.results[0].failure_code == ValidationFailureCode.MISSING_FILE


def test_validator_insufficient_size(tmp_path: Path) -> None:
    """Test validation failure for a file that is too small."""
    dataset_dir = tmp_path / "mock_data"
    dataset_dir.mkdir()

    file_path = dataset_dir / "data.txt"
    file_path.write_text("tiny")

    validator = DatasetValidator()
    validator.raw_dir = tmp_path

    constraints = (FileConstraint(id="c1", target_path="data.txt", min_size_bytes=100),)

    report = validator.validate("mock_data", "1.0", constraints)

    assert report.is_valid is False
    assert report.results[0].failure_code == ValidationFailureCode.INSUFFICIENT_SIZE


def test_validator_unreadable_file(tmp_path: Path) -> None:
    """Test validation failure for an unreadable file."""
    dataset_dir = tmp_path / "mock_data"
    dataset_dir.mkdir()

    file_path = dataset_dir / "data.txt"
    file_path.write_text("content")

    validator = DatasetValidator()
    validator.raw_dir = tmp_path

    constraints = (FileConstraint(id="c1", target_path="data.txt"),)

    # Mock os.access to return False for read permissions
    with patch("os.access", return_value=False):
        report = validator.validate("mock_data", "1.0", constraints)

    assert report.is_valid is False
    assert report.results[0].failure_code == ValidationFailureCode.UNREADABLE


def test_validator_corrupt_manifest(tmp_path: Path) -> None:
    """Test validation failure for malformed JSON manifest."""
    dataset_dir = tmp_path / "mock_data"
    dataset_dir.mkdir()

    manifest_path = dataset_dir / "manifest.json"
    manifest_path.write_text("{invalid_json")

    validator = DatasetValidator()
    validator.raw_dir = tmp_path

    constraints = (
        ManifestConstraint(id="c1", target_path="manifest.json", is_jsonl=False),
    )

    report = validator.validate("mock_data", "1.0", constraints)

    assert report.is_valid is False
    assert report.results[0].failure_code == ValidationFailureCode.MANIFEST_CORRUPT


def test_validator_jsonl_manifest(tmp_path: Path) -> None:
    """Test successful validation of JSONL manifest."""
    dataset_dir = tmp_path / "mock_data"
    dataset_dir.mkdir()

    manifest_path = dataset_dir / "manifest.jsonl"
    manifest_path.write_text('{"id": 1}\n{"id": 2}\n')

    validator = DatasetValidator()
    validator.raw_dir = tmp_path

    constraints = (
        ManifestConstraint(id="c1", target_path="manifest.jsonl", is_jsonl=True),
    )

    report = validator.validate("mock_data", "1.0", constraints)

    assert report.is_valid is True
