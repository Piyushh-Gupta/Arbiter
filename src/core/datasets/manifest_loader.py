"""Deterministic documentation loader for dataset manifests."""

import json

from pydantic import ValidationError

from src.core.constants import MANIFEST_FILENAME
from src.core.datasets.artifact_models import ArtifactIdentity
from src.core.datasets.manifest_models import DatasetManifest
from src.core.exceptions import (
    ManifestNotFoundError,
    ManifestParseError,
    UnsupportedSchemaVersionError,
)
from src.core.paths import ProjectPaths


class ManifestLoader:
    """
    Pure deterministic loader mapping physical manifest artifacts to immutable DatasetManifest structures.
    Decoupled from operational downloading and validation pipelines.
    """

    # The maximum major schema version this loader can currently process.
    SUPPORTED_MAJOR_VERSION = 1

    @classmethod
    def load_manifest(cls, identity: ArtifactIdentity) -> DatasetManifest:
        """
        Locate and load the manifest for a given canonical identity.

        Raises:
            ManifestNotFoundError: If the manifest file does not exist.
            ManifestParseError: If the manifest is malformed JSON or violates the schema.
            UnsupportedSchemaVersionError: If the manifest uses an unsupported schema version.
        """
        version_dir = ProjectPaths.get_dataset_version_dir(
            identity.dataset_id, identity.version
        )
        manifest_path = version_dir / MANIFEST_FILENAME

        if not manifest_path.exists():
            raise ManifestNotFoundError(
                f"Manifest not found for identity {identity.canonical}"
            )

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ManifestParseError(
                f"Malformed JSON in manifest {manifest_path}: {e}"
            ) from e
        except OSError as e:
            raise ManifestParseError(
                f"Failed to read manifest {manifest_path}: {e}"
            ) from e

        try:
            manifest = DatasetManifest.model_validate(data)
        except ValidationError as e:
            raise ManifestParseError(
                f"Schema validation failed for manifest {manifest_path}: {e}"
            ) from e

        if manifest.schema_version.major > cls.SUPPORTED_MAJOR_VERSION:
            raise UnsupportedSchemaVersionError(
                f"Unsupported manifest schema version {manifest.schema_version.version} "
                f"for identity {identity.canonical}. Max supported major version is {cls.SUPPORTED_MAJOR_VERSION}."
            )

        return manifest
