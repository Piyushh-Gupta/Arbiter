"""Custom exceptions for the Arbiter project."""


class ConfigurationError(Exception):
    """Raised when there is a configuration or environment error."""

    pass


class RegistryError(Exception):
    """Raised when there is an error interacting with the Dataset Registry."""

    pass
