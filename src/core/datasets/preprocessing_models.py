"""Data models for the preprocessing layer."""

import typing
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from src.core.datasets.mapping_models import TaskRecord
from src.core.datasets.partitioning_models import PartitionId


class PreprocessingMetadata(BaseModel):
    """Canonical structure storing preprocessing outputs and generic metadata."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


@dataclass(frozen=True)
class PreprocessedRecord:
    """Canonical structure wrapping records with accumulated preprocessing metadata."""

    partition: PartitionId
    record: TaskRecord
    preprocessing_metadata: PreprocessingMetadata


@typing.runtime_checkable
class PreprocessingDefinition(typing.Protocol):
    """Abstract protocol for declarative preprocessing parameters."""

    pass


class PassThroughPreprocessingDefinition(BaseModel):
    """Empty configuration driving the baseline pipeline proof of concept."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
