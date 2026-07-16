"""Project path definitions."""

from pathlib import Path

# Root directory is three levels up from this file (src/core/paths.py -> src/core -> src -> root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class ProjectPaths:
    """Centralized path definitions for the project."""

    ROOT: Path = PROJECT_ROOT

    # Data paths
    DATA_DIR: Path = ROOT / "data"
    DATA_RAW: Path = DATA_DIR / "raw"
    DATA_PROCESSED: Path = DATA_DIR / "processed"
    DATA_INDEX: Path = DATA_DIR / "index"

    # Other primary directories
    CONFIG_DIR: Path = ROOT / "configs"
    ARTIFACTS_DIR: Path = ROOT / "artifacts"
    LOGS_DIR: Path = ROOT / "logs"

    @classmethod
    def resolve_path(cls, path_str: str) -> Path:
        """Resolve a configuration path string relative to the project root."""
        return cls.ROOT / path_str

    @classmethod
    def get_dataset_version_dir(cls, dataset_id: str, version: str) -> Path:
        """Deterministically resolve the isolated storage directory for a specific dataset version."""
        return cls.DATA_RAW / dataset_id / version

    @classmethod
    def get_all_required_dirs(cls) -> list[Path]:
        """Return a list of all required directories."""
        return [
            cls.DATA_RAW,
            cls.DATA_PROCESSED,
            cls.DATA_INDEX,
            cls.CONFIG_DIR,
            cls.ARTIFACTS_DIR,
            cls.LOGS_DIR,
        ]
