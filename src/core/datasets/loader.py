"""Deterministic IO loader for physical dataset artifacts."""

import structlog

from src.core.datasets.artifact_manager import ArtifactManager
from src.core.datasets.artifact_models import ArtifactIdentity, ArtifactLifecycleState
from src.core.datasets.loader_models import ArtifactHandle
from src.core.datasets.manifest_models import ArtifactInventoryEntry
from src.core.exceptions import (
    ArtifactNotFoundError,
    ArtifactNotReadyError,
    PathTraversalError,
    UnreadableArtifactError,
)

logger = structlog.get_logger(__name__)


class DatasetLoader:
    """
    Instance-based pure IO loader service for physical dataset artifacts.
    Strictly responsible for safely opening resolved files as binary streams.
    Has no awareness of logical splits or encodings.
    """

    def __init__(self) -> None:
        pass  # Reserved for future dependency injection

    def open_artifact(
        self, identity: ArtifactIdentity, entry: ArtifactInventoryEntry
    ) -> ArtifactHandle:
        """
        Opens a binary stream for a specific, pre-resolved artifact documented in the manifest's inventory.

        Args:
            identity: The canonical dataset identity.
            entry: The pre-resolved inventory entry detailing the physical filename.

        Returns:
            An ArtifactHandle wrapping the open binary IO stream and the entry.

        Raises:
            ArtifactNotReadyError: If the dataset is not in the READY lifecycle state.
            PathTraversalError: If the resolved path attempts to escape the version directory.
            ArtifactNotFoundError: If the physical file does not exist on disk.
            UnreadableArtifactError: If the file cannot be opened (e.g. permission error).
        """
        # 1. Verify artifact lifecycle state
        artifact = ArtifactManager.resolve_artifact(identity)
        if artifact.state != ArtifactLifecycleState.READY:
            logger.error(
                "Attempted to load non-READY artifact",
                canonical=identity.canonical,
                state=artifact.state.value,
            )
            raise ArtifactNotReadyError(
                f"Dataset {identity.canonical} cannot be loaded. Current state: {artifact.state.value}"
            )

        # 2. Secure path resolution
        base_dir = artifact.path.resolve()
        target_path = (base_dir / entry.filename).resolve()

        # Ensure the resolved target path is strictly a descendant of the base directory
        # This prevents symlink escapes and path traversal (e.g., "../../etc/passwd")
        if not target_path.is_relative_to(base_dir):
            logger.error(
                "Path traversal attempt detected",
                canonical=identity.canonical,
                filename=entry.filename,
                resolved_path=str(target_path),
            )
            raise PathTraversalError(
                f"Artifact filename '{entry.filename}' resolves outside the isolated version directory."
            )

        # 3. Target Verification
        if not target_path.exists() or not target_path.is_file():
            logger.error(
                "Artifact file not found on disk",
                canonical=identity.canonical,
                filename=entry.filename,
            )
            raise ArtifactNotFoundError(
                f"Physical file for artifact '{entry.filename}' not found."
            )

        # 4. Execution
        try:
            stream = open(target_path, "rb")
        except OSError as e:
            logger.error(
                "Failed to read artifact",
                canonical=identity.canonical,
                filename=entry.filename,
                error=str(e),
            )
            raise UnreadableArtifactError(
                f"Failed to open artifact '{entry.filename}' for reading: {e}"
            ) from e

        logger.info(
            "Artifact successfully opened",
            canonical=identity.canonical,
            filename=entry.filename,
        )

        return ArtifactHandle(stream=stream, identity=identity, entry=entry)
