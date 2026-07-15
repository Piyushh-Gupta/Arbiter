"""Dataset abstraction and registry layer."""

from src.core.datasets.metadata import DatasetMetadata, DatasetSchema, DatasetSplit
from src.core.datasets.registry import registry

__all__ = [
    "DatasetMetadata",
    "DatasetSchema",
    "DatasetSplit",
    "registry",
]
