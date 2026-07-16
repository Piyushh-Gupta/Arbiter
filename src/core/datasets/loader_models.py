"""Models for representing securely opened dataset artifacts."""

import typing

from src.core.datasets.artifact_models import ArtifactIdentity
from src.core.datasets.manifest_models import ArtifactInventoryEntry


class ArtifactHandle:
    """
    A context-managed wrapper owning a read-only binary stream and
    the associated artifact metadata from the manifest.

    This handle explicitly avoids implementing __del__ for cleanup,
    relying entirely on deterministic context management by the caller.
    """

    def __init__(
        self,
        stream: typing.BinaryIO,
        identity: ArtifactIdentity,
        entry: ArtifactInventoryEntry,
    ) -> None:
        self._stream = stream
        self._identity = identity
        self._entry = entry

    @property
    def identity(self) -> ArtifactIdentity:
        """The canonical identity of the dataset."""
        return self._identity

    @property
    def entry(self) -> ArtifactInventoryEntry:
        """The manifest entry describing this specific artifact."""
        return self._entry

    def __enter__(self) -> typing.BinaryIO:
        """Enter the context manager, yielding the raw binary stream."""
        return self._stream

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: typing.Any | None,
    ) -> None:
        """Ensure deterministic cleanup of the underlying file descriptor."""
        self._stream.close()
