"""Unit tests for the configuration layer."""

from src.core.bootstrap import initialize_application
from src.core.config import Settings
from src.core.paths import ProjectPaths
from src.core.validation import validate_environment, validate_paths


def test_paths_definition() -> None:
    """Test that project paths are defined relative to the expected root."""
    # The root should be Arbiter directory (or whatever the root is named)
    assert ProjectPaths.DATA_DIR.name == "data"
    assert ProjectPaths.CONFIG_DIR.name == "configs"
    assert ProjectPaths.ARTIFACTS_DIR.name == "artifacts"


def test_settings_default_values() -> None:
    """Test that default settings are loaded correctly."""
    settings = Settings()
    assert settings.app_name == "Arbiter API"
    assert settings.environment in ("development", "test", "staging", "production")
    assert settings.log.level == "INFO"
    assert settings.paths is ProjectPaths


def test_validate_environment_valid() -> None:
    """Test environment validation passes for default environment."""
    # Should not raise ConfigurationError
    validate_environment()


def test_validate_paths_valid() -> None:
    """Test path validation passes for valid root."""
    # Should not raise ConfigurationError
    validate_paths()


def test_initialize_application() -> None:
    """Test that bootstrap runs without errors and creates required directories."""
    # Ensure it creates the directories and validates without raising
    initialize_application()
    for directory in ProjectPaths.get_all_required_dirs():
        assert directory.exists()
        assert directory.is_dir()
