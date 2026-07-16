"""Stateless filter implementations operating via generic selectors."""

from collections.abc import Iterator

from src.core.datasets.filtering_models import (
    FieldEqualsPredicate,
    FieldExistsPredicate,
    FieldInSetPredicate,
    FieldLengthPredicate,
)
from src.core.datasets.mapping_models import TaskRecord


class FieldExistsFilter:
    """Filter ensuring a target field exists and is not None."""

    def filter_stream(
        self, stream: Iterator[TaskRecord], definition: FieldExistsPredicate
    ) -> Iterator[TaskRecord]:
        for record in stream:
            value = definition.selector.resolve(record)
            if value is not None:
                yield record


class FieldEqualsFilter:
    """Filter ensuring a target field exactly equals a value."""

    def filter_stream(
        self, stream: Iterator[TaskRecord], definition: FieldEqualsPredicate
    ) -> Iterator[TaskRecord]:
        for record in stream:
            value = definition.selector.resolve(record)
            if value == definition.target_value:
                yield record


class FieldInSetFilter:
    """Filter ensuring a target field is within a predefined subset."""

    def filter_stream(
        self, stream: Iterator[TaskRecord], definition: FieldInSetPredicate
    ) -> Iterator[TaskRecord]:
        for record in stream:
            value = definition.selector.resolve(record)
            if value in definition.allowed_values:
                yield record


class FieldLengthFilter:
    """Filter enforcing character length boundaries on string fields."""

    def filter_stream(
        self, stream: Iterator[TaskRecord], definition: FieldLengthPredicate
    ) -> Iterator[TaskRecord]:
        for record in stream:
            value = definition.selector.resolve(record)
            if value is None:
                continue

            # Length predicates generally apply to strings or iterables
            if not hasattr(value, "__len__"):
                continue

            length = len(value)

            if definition.min_length is not None and length < definition.min_length:
                continue

            if definition.max_length is not None and length > definition.max_length:
                continue

            yield record
