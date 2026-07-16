"""Abstract base classes and protocols for the schema mapping layer."""

import typing
from collections.abc import Iterator
from dataclasses import dataclass

from src.core.datasets.mapping_models import SchemaDefinition, TaskRecord
from src.core.datasets.normalization_models import NormalizedRecord


class BaseMapper(typing.Protocol):
    """Protocol for executing a schema mapping strategy."""

    def map_stream(
        self, stream: Iterator[NormalizedRecord], schema_def: SchemaDefinition
    ) -> Iterator[TaskRecord]:
        """
        Executes the mapping strategy.

        Args:
            stream: A lazy stream of canonical NormalizedRecords.
            schema_def: The immutable mapping configuration.

        Yields:
            Immutable TaskRecord instances mapped from the normalized inputs.
        """
        ...


@dataclass(frozen=True)
class TaskMapping:
    """
    Unified, immutable representation of a resolved mapping configuration.

    Couples exactly one declarative SchemaDefinition with its corresponding
    execution mapper strategy.
    """

    schema_definition: SchemaDefinition
    mapper: BaseMapper
