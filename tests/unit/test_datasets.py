"""Unit tests for the Dataset Registry."""

import pytest
from pydantic import ValidationError

from src.core.datasets.metadata import DatasetMetadata, DatasetSchema, DatasetSplit
from src.core.datasets.registry import DatasetRegistry
from src.core.exceptions import RegistryError


def test_dataset_metadata_immutability() -> None:
    """Test that metadata models are immutable."""
    schema = DatasetSchema(features=("text",), target_column="label")
    metadata = DatasetMetadata(
        id="test_dataset",
        version="1.0.0",
        description="A test dataset",
        domain="testing",
        schema_metadata=schema,
        splits=(DatasetSplit.TRAIN, DatasetSplit.TEST),
    )

    with pytest.raises(ValidationError):
        metadata.id = "new_id"

    with pytest.raises(ValidationError):
        schema.features = ("new_feature",)


def test_registry_register_and_get() -> None:
    """Test basic register and get functionality."""
    registry = DatasetRegistry()
    metadata = DatasetMetadata(
        id="test_data",
        version="1.0",
        description="Test",
        domain="Test",
        schema_metadata=DatasetSchema(),
        splits=(DatasetSplit.TRAIN,),
    )

    registry.register_dataset(metadata)

    retrieved = registry.get_dataset("test_data", "1.0")
    assert retrieved is metadata


def test_registry_duplicate_registration() -> None:
    """Test that duplicate registrations fail."""
    registry = DatasetRegistry()
    metadata = DatasetMetadata(
        id="test_data",
        version="1.0",
        description="Test",
        domain="Test",
        schema_metadata=DatasetSchema(),
        splits=(DatasetSplit.TRAIN,),
    )

    registry.register_dataset(metadata)

    with pytest.raises(RegistryError):
        registry.register_dataset(metadata)


def test_registry_get_unknown() -> None:
    """Test that getting an unknown dataset fails fast."""
    registry = DatasetRegistry()
    with pytest.raises(RegistryError):
        registry.get_dataset("unknown", "1.0")


def test_registry_list_datasets() -> None:
    """Test listing registered datasets."""
    registry = DatasetRegistry()
    metadata = DatasetMetadata(
        id="test_data",
        version="1.0",
        description="Test",
        domain="Test",
        schema_metadata=DatasetSchema(),
        splits=(DatasetSplit.TRAIN,),
    )
    registry.register_dataset(metadata)

    datasets = registry.list_datasets()
    assert len(datasets) == 1
    assert datasets[0].id == "test_data"
