"""Unit tests for the M2.1 Dataset Loading Layer."""

from pathlib import Path
from unittest.mock import patch

import pytest

from src.core.datasets.artifact_manager import ArtifactManager
from src.core.datasets.artifact_models import ArtifactIdentity
from src.core.datasets.loader import DatasetLoader
from src.core.datasets.manifest_models import ArtifactInventoryEntry
from src.core.exceptions import (
    ArtifactNotFoundError,
    ArtifactNotReadyError,
    PathTraversalError,
    UnreadableArtifactError,
)
from src.core.paths import ProjectPaths


@pytest.fixture
def mock_identity() -> ArtifactIdentity:
    return ArtifactIdentity(dataset_id="test_dataset", version="1.0.0")


@pytest.fixture
def mock_entry() -> ArtifactInventoryEntry:
    return ArtifactInventoryEntry(
        filename="train_data.jsonl",
        file_type="jsonl",
        role="train",
        description="Training split data",
    )


def setup_artifact_dir(identity: ArtifactIdentity, tmp_path: Path) -> Path:
    ProjectPaths.DATA_RAW = tmp_path
    version_dir = ProjectPaths.get_dataset_version_dir(
        identity.dataset_id, identity.version
    )
    version_dir.mkdir(parents=True, exist_ok=True)
    return version_dir


def test_dataset_loader_success(
    mock_identity: ArtifactIdentity, mock_entry: ArtifactInventoryEntry, tmp_path: Path
) -> None:
    """Test successful loading of a READY artifact yields an open binary stream."""
    version_dir = setup_artifact_dir(mock_identity, tmp_path)

    # Mark as READY
    ArtifactManager.mark_ready(mock_identity)

    # Create the physical file
    target_file = version_dir / mock_entry.filename
    target_file.write_bytes(b"test data")

    loader = DatasetLoader()
    handle = loader.open_artifact(mock_identity, mock_entry)

    assert handle.identity == mock_identity
    assert handle.entry == mock_entry

    with handle as stream:
        assert not stream.closed
        assert stream.read() == b"test data"

    assert stream.closed


def test_dataset_loader_not_ready(
    mock_identity: ArtifactIdentity, mock_entry: ArtifactInventoryEntry, tmp_path: Path
) -> None:
    """Test loading fails if the artifact is not READY."""
    setup_artifact_dir(mock_identity, tmp_path)

    # Dataset is inherently UNINITIALIZED or DOWNLOADED, but not READY
    loader = DatasetLoader()

    with pytest.raises(ArtifactNotReadyError, match="cannot be loaded. Current state:"):
        loader.open_artifact(mock_identity, mock_entry)


def test_dataset_loader_artifact_not_found(
    mock_identity: ArtifactIdentity, mock_entry: ArtifactInventoryEntry, tmp_path: Path
) -> None:
    """Test loading fails if the requested physical file does not exist."""
    setup_artifact_dir(mock_identity, tmp_path)
    ArtifactManager.mark_ready(mock_identity)

    loader = DatasetLoader()

    # File is never created on disk
    with pytest.raises(ArtifactNotFoundError, match="not found"):
        loader.open_artifact(mock_identity, mock_entry)


def test_dataset_loader_path_traversal_protection(
    mock_identity: ArtifactIdentity, tmp_path: Path
) -> None:
    """Test loading prevents symlink escapes or path traversal attempts."""
    setup_artifact_dir(mock_identity, tmp_path)
    ArtifactManager.mark_ready(mock_identity)

    # Malicious entry attempting to escape the dataset directory
    malicious_entry = ArtifactInventoryEntry(
        filename="../../etc/passwd",
        file_type="txt",
        role="malicious",
        description="Path traversal attempt",
    )

    loader = DatasetLoader()

    with pytest.raises(
        PathTraversalError, match="resolves outside the isolated version directory"
    ):
        loader.open_artifact(mock_identity, malicious_entry)


def test_dataset_loader_unreadable_artifact(
    mock_identity: ArtifactIdentity, mock_entry: ArtifactInventoryEntry, tmp_path: Path
) -> None:
    """Test loading catches OSErrors (e.g. permission issues) cleanly."""
    version_dir = setup_artifact_dir(mock_identity, tmp_path)
    ArtifactManager.mark_ready(mock_identity)

    target_file = version_dir / mock_entry.filename
    target_file.write_bytes(b"test data")

    loader = DatasetLoader()

    # Mock built-in open to raise an OSError
    with patch("builtins.open", side_effect=PermissionError("Permission denied")):
        with pytest.raises(UnreadableArtifactError, match="Failed to open artifact"):
            loader.open_artifact(mock_identity, mock_entry)


def test_dataset_loader_lazy_loading(
    mock_identity: ArtifactIdentity, mock_entry: ArtifactInventoryEntry, tmp_path: Path
) -> None:
    """Test that open_artifact does not eagerly read from the stream."""
    version_dir = setup_artifact_dir(mock_identity, tmp_path)
    ArtifactManager.mark_ready(mock_identity)

    target_file = version_dir / mock_entry.filename
    target_file.write_bytes(b"test data")

    loader = DatasetLoader()

    with patch("builtins.open") as mock_open:
        _handle = loader.open_artifact(mock_identity, mock_entry)

        # Open should be called to get the descriptor
        mock_open.assert_called_once()

        # But read should never be called until the stream is used
        mock_stream = mock_open.return_value
        mock_stream.read.assert_not_called()
