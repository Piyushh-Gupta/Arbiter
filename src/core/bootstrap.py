"""Application bootstrap and initialization routines."""

from src.core.paths import ProjectPaths
from src.core.validation import validate_startup


def _create_required_directories() -> None:
    """Explicitly create all required directories defined in paths."""
    for directory in ProjectPaths.get_all_required_dirs():
        directory.mkdir(parents=True, exist_ok=True)


def initialize_application() -> None:
    """
    Execute the startup orchestration routine.

    This function should be called at the very beginning of the application lifecycle.
    It coordinates directory creation and system validation.
    """
    _create_required_directories()
    validate_startup()
