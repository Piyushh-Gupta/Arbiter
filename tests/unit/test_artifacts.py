"""Unit tests for dataset artifacts and version management."""

from pathlib import Path

import pytest

from src.core.datasets.artifact_manager import ArtifactManager
from src.core.datasets.artifact_models import ArtifactIdentity, ArtifactLifecycleState
from src.core.datasets.metadata import DatasetMetadata, DatasetSchema, DatasetSplit
from src.core.datasets.registry import registry
from src.core.exceptions import InvalidIdentityError, VersionNotFoundError
from src.core.paths import ProjectPaths


@pytest.fixture
def mock_dataset_identity() -> ArtifactIdentity:
    identity = ArtifactIdentity(dataset_id="test_dataset", version="1.0.0")

    # Register mock metadata
    try:
        registry.register_dataset(
            DatasetMetadata(
                id=identity.dataset_id,
                version=identity.version,
                description="Mock dataset",
                domain="mock",
                schema_metadata=DatasetSchema(),
                splits=(DatasetSplit.TRAIN,),
            )
        )
    except Exception:
        pass  # Already registered

    return identity


def test_artifact_identity_canonical() -> None:
    """Test identity canonical string and validation."""
    identity = ArtifactIdentity(dataset_id="mock", version="2.0.5-alpha")
    assert identity.canonical == "mock@2.0.5-alpha"

    with pytest.raises(InvalidIdentityError):
        ArtifactIdentity(dataset_id="mock", version="../../etc/passwd")


def test_artifact_manager_uninitialized(
    mock_dataset_identity: ArtifactIdentity, tmp_path: Path
) -> None:
    """Test resolution of an uninitialized dataset."""
    ProjectPaths.DATA_RAW = tmp_path

    artifact = ArtifactManager.resolve_artifact(mock_dataset_identity)

    assert artifact.identity == mock_dataset_identity
    assert artifact.state == ArtifactLifecycleState.UNINITIALIZED
    assert not artifact.path.exists()


def test_artifact_manager_incomplete(
    mock_dataset_identity: ArtifactIdentity, tmp_path: Path
) -> None:
    """Test resolution of an incomplete dataset (has .part files)."""
    ProjectPaths.DATA_RAW = tmp_path

    version_dir = ProjectPaths.get_dataset_version_dir(
        mock_dataset_identity.dataset_id, mock_dataset_identity.version
    )
    version_dir.mkdir(parents=True)
    (version_dir / "data.part").touch()

    artifact = ArtifactManager.resolve_artifact(mock_dataset_identity)

    assert artifact.state == ArtifactLifecycleState.INCOMPLETE


def test_artifact_manager_downloaded(
    mock_dataset_identity: ArtifactIdentity, tmp_path: Path
) -> None:
    """Test resolution of a downloaded but unvalidated dataset."""
    ProjectPaths.DATA_RAW = tmp_path

    version_dir = ProjectPaths.get_dataset_version_dir(
        mock_dataset_identity.dataset_id, mock_dataset_identity.version
    )
    version_dir.mkdir(parents=True)
    (version_dir / "data.json").touch()

    artifact = ArtifactManager.resolve_artifact(mock_dataset_identity)

    assert artifact.state == ArtifactLifecycleState.DOWNLOADED


def test_artifact_manager_ready(
    mock_dataset_identity: ArtifactIdentity, tmp_path: Path
) -> None:
    """Test resolution of a validated and ready dataset."""
    ProjectPaths.DATA_RAW = tmp_path

    version_dir = ProjectPaths.get_dataset_version_dir(
        mock_dataset_identity.dataset_id, mock_dataset_identity.version
    )
    version_dir.mkdir(parents=True)
    (version_dir / "data.json").touch()

    ArtifactManager.mark_ready(mock_dataset_identity)

    artifact = ArtifactManager.resolve_artifact(mock_dataset_identity)

    assert artifact.state == ArtifactLifecycleState.READY


def test_artifact_manager_corrupted(
    mock_dataset_identity: ArtifactIdentity, tmp_path: Path
) -> None:
    """Test resolution of a corrupted dataset."""
    ProjectPaths.DATA_RAW = tmp_path

    version_dir = ProjectPaths.get_dataset_version_dir(
        mock_dataset_identity.dataset_id, mock_dataset_identity.version
    )
    version_dir.mkdir(parents=True)
    (version_dir / "data.json").touch()

    ArtifactManager.mark_corrupted(mock_dataset_identity)

    artifact = ArtifactManager.resolve_artifact(mock_dataset_identity)

    assert artifact.state == ArtifactLifecycleState.CORRUPTED


def test_artifact_manager_unregistered_identity() -> None:
    """Test resolution fails if the dataset is not registered."""
    identity = ArtifactIdentity(dataset_id="ghost", version="1.0.0")

    with pytest.raises(VersionNotFoundError):
        ArtifactManager.resolve_artifact(identity)
