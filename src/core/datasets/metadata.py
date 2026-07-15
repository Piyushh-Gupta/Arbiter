"""Immutable metadata models for dataset registration."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class DatasetSplit(Enum):
    """Strongly typed immutable dataset splits."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class DatasetSchema(BaseModel):
    """Strongly typed model defining the expected structure of a dataset."""

    features: tuple[str, ...] = Field(default_factory=tuple)
    target_column: Optional[str] = None

    model_config = ConfigDict(frozen=True)


class DatasetMetadata(BaseModel):
    """Immutable data model representing dataset metadata."""

    id: str = Field(
        ..., description="Unique identifier for the dataset (e.g., 'fever')"
    )
    version: str = Field(..., description="Semantic version or snapshot identifier")
    description: str = Field(..., description="Human-readable purpose of the dataset")
    domain: str = Field(..., description="Origin or domain categorization")
    schema_metadata: DatasetSchema = Field(...)
    splits: tuple[DatasetSplit, ...] = Field(default_factory=tuple)

    model_config = ConfigDict(frozen=True)
