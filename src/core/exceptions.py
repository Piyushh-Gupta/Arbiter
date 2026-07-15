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
