"""Stateless coordination layer for dataset artifacts and version resolution."""

from pathlib import Path

from src.core.datasets.artifact_models import (
    ArtifactIdentity,
    ArtifactLifecycleState,
    DatasetArtifact,
)
from src.core.datasets.registry import registry
from src.core.exceptions import RegistryError, VersionNotFoundError
from src.core.paths import ProjectPaths


class ArtifactManager:
    """
    Stateless coordinator for canonical identity resolution, deriving physical storage paths,
    and computing lifecycle states dynamically by inspecting the filesystem and registry.
    """

    @staticmethod
    def _has_part_files(directory: Path) -> bool:
        """Check if any .part files exist in the directory (recursive)."""
        # A quick check for .part files to signify incomplete downloads.
        # We only check top level for simplicity, assuming flat dataset structure or downloading to root.
        try:
            return any(f.suffix == ".part" for f in directory.iterdir())
        except OSError:
            return False

    @classmethod
    def resolve_artifact(cls, identity: ArtifactIdentity) -> DatasetArtifact:
        """
        Derive the current artifact state deterministically from the registry and filesystem.
        """
        # Ensure it exists in the registry to prevent working with ghost datasets
        try:
            registry.get_dataset(identity.dataset_id, identity.version)
        except RegistryError as e:
            raise VersionNotFoundError(
                f"Identity {identity.canonical} not registered."
            ) from e

        version_dir = ProjectPaths.get_dataset_version_dir(
            identity.dataset_id, identity.version
        )

        # Derive state
        if not version_dir.exists():
            state = ArtifactLifecycleState.UNINITIALIZED
        elif cls._has_part_files(version_dir):
            state = ArtifactLifecycleState.INCOMPLETE
        elif (version_dir / ".corrupted").exists():
            state = ArtifactLifecycleState.CORRUPTED
        elif (version_dir / ".validated").exists():
            state = ArtifactLifecycleState.READY
        else:
            state = ArtifactLifecycleState.DOWNLOADED

        return DatasetArtifact(
            identity=identity,
            state=state,
            path=version_dir,
        )

    @classmethod
    def mark_ready(cls, identity: ArtifactIdentity) -> None:
        """Drop a sentinel file to transition state to READY."""
        version_dir = ProjectPaths.get_dataset_version_dir(
            identity.dataset_id, identity.version
        )
        if version_dir.exists():
            (version_dir / ".validated").touch()
            # Clear corrupted marker if it somehow existed
            corrupted_marker = version_dir / ".corrupted"
            if corrupted_marker.exists():
                corrupted_marker.unlink()

    @classmethod
    def mark_corrupted(cls, identity: ArtifactIdentity) -> None:
        """Drop a sentinel file to transition state to CORRUPTED."""
        version_dir = ProjectPaths.get_dataset_version_dir(
            identity.dataset_id, identity.version
        )
        if version_dir.exists():
            (version_dir / ".corrupted").touch()
            # Clear validated marker if it somehow existed
            validated_marker = version_dir / ".validated"
            if validated_marker.exists():
                validated_marker.unlink()
