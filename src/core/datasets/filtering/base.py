"""Abstract base protocols for the filtering layer."""

import typing
from collections.abc import Iterator

from src.core.datasets.mapping_models import TaskRecord


@typing.runtime_checkable
class BaseFilter(typing.Protocol):
    """Execution-only protocol consuming an incoming stream and producing a filtered stream."""

    def filter_stream(
        self, stream: Iterator[TaskRecord], definition: typing.Any
    ) -> Iterator[TaskRecord]:
        """
        Executes the filter based on the declarative definition.

        Args:
            stream: A lazy stream of immutable TaskRecords.
            definition: The declarative intent (predicate model) to evaluate.

        Yields:
            Immutable TaskRecords that pass the predicate evaluation.
        """
        ...
