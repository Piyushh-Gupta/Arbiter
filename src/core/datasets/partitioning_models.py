"""Data models and predicates for the partitioning layer."""

import typing
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from src.core.datasets.mapping_models import TaskRecord
from src.core.datasets.selectors import FieldSelector


@dataclass(frozen=True)
class PartitionId:
    """Immutable value object encapsulating partition identity."""

    name: str


@typing.runtime_checkable
class PartitionDefinition(typing.Protocol):
    """Abstract protocol for declarative partition configurations."""

    pass


class ModuloPartitionDefinition(BaseModel):
    """Declarative config assigning records via integer modulo arithmetic."""

    selector: FieldSelector
    modulo_divisor: int
    partition_map: dict[PartitionId, frozenset[int]]

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class HashPartitionDefinition(BaseModel):
    """Declarative config assigning records via deterministic hashing."""

    selector: FieldSelector
    hash_buckets: int
    partition_map: dict[PartitionId, frozenset[int]]

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


@dataclass(frozen=True)
class PartitionedRecord:
    """Immutable container wrapping a task record and its assigned logical partition."""

    partition: PartitionId
    record: TaskRecord
