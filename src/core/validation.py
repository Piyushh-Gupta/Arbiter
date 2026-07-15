"""Centralized validation routines for configuration and environment."""

from src.core.config import settings
from src.core.exceptions import ConfigurationError
from src.core.paths import ProjectPaths


def validate_environment() -> None:
    """Validate that required environment variables and settings are sane."""
    valid_envs = ("development", "staging", "production", "test")
    if settings.environment not in valid_envs:
        raise ConfigurationError(
            f"Invalid environment: {settings.environment}. Must be one of {valid_envs}"
        )


def validate_paths() -> None:
    """Validate that required paths are accessible or can be created."""
    if not ProjectPaths.ROOT.exists():
        raise ConfigurationError(f"Project root does not exist: {ProjectPaths.ROOT}")


def validate_configuration() -> None:
    """Validate cross-cutting configuration constraints."""
    # Ensure active dataset is registered
    if settings.dataset.id is not None:
        from src.core.datasets import registry
        from src.core.exceptions import RegistryError

        # We need a version to look up, or we assume a default.
        # If version is required, config should provide it.
        version = settings.dataset.version or "1.0.0"
        try:
            registry.get_dataset(settings.dataset.id, version)
        except RegistryError as e:
            raise ConfigurationError(
                f"Active dataset {settings.dataset.id}@{version} is not registered."
            ) from e

    # Validate download settings
    if settings.download.timeout_seconds <= 0:
        raise ConfigurationError("Download timeout must be strictly positive.")


def validate_startup() -> None:
    """Run all validation routines required for startup."""
    validate_environment()
    validate_paths()
    validate_configuration()
