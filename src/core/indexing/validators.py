import hashlib
import os

from src.core.exceptions import RetrievalConfigurationError
from src.core.indexing.models import IndexManifest
from src.core.indexing.pipeline import ArtifactValidator


def _hash_file(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class ManifestArtifactValidator(ArtifactValidator):
    """Validates that all artifacts defined in the manifest exist and match checksums."""

    def __init__(self, expected_dataset_version: str, expected_dimension: int) -> None:
        self.expected_dataset_version = expected_dataset_version
        self.expected_dimension = expected_dimension

    def validate(self, manifest: IndexManifest) -> None:
        if manifest.dataset_version != self.expected_dataset_version:
            raise RetrievalConfigurationError(
                f"Manifest dataset version {manifest.dataset_version} does not match expected {self.expected_dataset_version}"
            )

        if manifest.embedding_dimension != self.expected_dimension:
            raise RetrievalConfigurationError(
                f"Manifest embedding dimension {manifest.embedding_dimension} does not match expected {self.expected_dimension}"
            )

        for name, artifact in manifest.artifacts.items():
            if not os.path.exists(artifact.path):
                raise RetrievalConfigurationError(
                    f"Artifact {name} not found at {artifact.path}"
                )

            actual_checksum = _hash_file(artifact.path)
            if actual_checksum != artifact.checksum:
                raise RetrievalConfigurationError(
                    f"Checksum mismatch for artifact {name}. Expected {artifact.checksum}, got {actual_checksum}"
                )
