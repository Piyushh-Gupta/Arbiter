"""Abstract base protocols and mapping for the partitioning layer."""

import typing
from dataclasses import dataclass

from src.core.datasets.mapping_models import TaskRecord
from src.core.datasets.partitioning_models import (
    ModuloPartitionDefinition,
    PartitionDefinition,
    PartitionId,
)
from src.core.exceptions import PartitionConfigurationError


@typing.runtime_checkable
class BasePartitioner(typing.Protocol):
    """Stateless protocol for determining partition assignment."""

    def assign(
        self, record: TaskRecord, definition: PartitionDefinition
    ) -> PartitionId | None: ...


@dataclass(frozen=True)
class PartitionMapping:
    """Unified configuration wrapping the declarative intent and execution strategy."""

    definition: PartitionDefinition
    strategy: BasePartitioner

    def __post_init__(self) -> None:
        """Enforce strategy-to-definition configuration compatibility explicitly before hot-path execution."""
        strategy_name = type(self.strategy).__name__

        if strategy_name == "ModuloPartitioner" and not isinstance(
            self.definition, ModuloPartitionDefinition
        ):
            raise PartitionConfigurationError(
                "ModuloPartitioner explicitly requires a ModuloPartitionDefinition."
            )

        # Future configurations follow the identical rigid binding structure.
