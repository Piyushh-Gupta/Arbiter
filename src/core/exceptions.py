"""Custom exceptions for the Arbiter project."""


class ConfigurationError(Exception):
    """Raised when there is a configuration or environment error."""

    pass


class RegistryError(Exception):
    """Raised when there is an error interacting with the Dataset Registry."""

    pass


class DownloadError(Exception):
    """Raised when an error occurs during artifact download."""

    pass


class IntegrityError(Exception):
    """Raised when an artifact fails cryptographic integrity verification."""

    pass


class DatasetValidationError(Exception):
    """Raised by the orchestration layer when a dataset fails validation."""

    pass


class VersionNotFoundError(Exception):
    """Raised when a requested dataset version cannot be resolved."""

    pass


class InvalidIdentityError(Exception):
    """Raised when an artifact identity string is malformed or invalid."""

    pass


class ManifestNotFoundError(Exception):
    """Raised when a requested dataset manifest cannot be found."""

    pass


class ManifestParseError(Exception):
    """Raised when a manifest file is malformed or fails schema validation."""

    pass


class UnsupportedSchemaVersionError(Exception):
    """Raised when a manifest uses an unsupported schema version."""

    pass


class ArtifactNotReadyError(Exception):
    """Raised when attempting to load an artifact that is not in the READY lifecycle state."""

    pass


class ArtifactNotFoundError(Exception):
    """Raised when a physical artifact file cannot be found on disk."""

    pass


class UnreadableArtifactError(Exception):
    """Raised when an artifact exists but cannot be read (e.g., permission error, corrupted handle)."""

    pass


class PathTraversalError(Exception):
    """Raised when an artifact filename attempts to escape the isolated version directory."""

    pass
