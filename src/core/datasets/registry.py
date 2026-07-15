"""Passive centralized registry for known datasets."""

from src.core.datasets.metadata import DatasetMetadata
from src.core.exceptions import RegistryError


class DatasetRegistry:
    """
    A strictly typed, passive encyclopedia of registered datasets.

    This class does not manage active configurations or filesystem operations.
    It purely maps dataset identifiers to their immutable metadata schemas.
    """

    def __init__(self) -> None:
        # Maps f"{id}@{version}" to DatasetMetadata
        self._datasets: dict[str, DatasetMetadata] = {}

    def _get_key(self, dataset_id: str, version: str) -> str:
        return f"{dataset_id}@{version}"

    def register_dataset(self, metadata: DatasetMetadata) -> None:
        """Register a dataset into the encyclopedia."""
        key = self._get_key(metadata.id, metadata.version)
        if key in self._datasets:
            raise RegistryError(f"Dataset already registered: {key}")

        self._datasets[key] = metadata

    def get_dataset(self, dataset_id: str, version: str) -> DatasetMetadata:
        """Retrieve dataset metadata by id and version."""
        key = self._get_key(dataset_id, version)
        if key not in self._datasets:
            raise RegistryError(f"Dataset not found in registry: {key}")

        return self._datasets[key]

    def list_datasets(self) -> list[DatasetMetadata]:
        """List all registered datasets."""
        return list(self._datasets.values())


# Global singleton instance of the passive registry
registry = DatasetRegistry()
