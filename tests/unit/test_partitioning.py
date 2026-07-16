"""Unit tests for the Dataset Partitioning Layer (M2.6)."""

import typing
from collections.abc import Iterator

import pytest

from src.core.datasets.filtering_models import SimpleFieldSelector
from src.core.datasets.mapping_models import ClassificationRecord, TaskRecord
from src.core.datasets.normalization_models import ProvenanceMetadata
from src.core.datasets.partitioner import DatasetPartitioner
from src.core.datasets.partitioning.base import PartitionMapping
from src.core.datasets.partitioning.implementations import (
    HashPartitioner,
    ModuloPartitioner,
)
from src.core.datasets.partitioning_models import (
    HashPartitionDefinition,
    ModuloPartitionDefinition,
    PartitionedRecord,
    PartitionId,
)
from src.core.exceptions import (
    PartitionAssignmentError,
    PartitionConfigurationError,
    PartitionExecutionError,
)


def _mock_classification_stream(count: int = 5) -> Iterator[TaskRecord]:
    for i in range(1, count + 1):
        yield ClassificationRecord(
            text=f"Sample text {i}",
            label="SUPPORTS" if i % 2 == 0 else "REFUTES",
            provenance=ProvenanceMetadata(record_index=i),
        )


def _mock_classification_stream_with_ids(count: int = 5) -> Iterator[TaskRecord]:
    for i in range(1, count + 1):
        yield ClassificationRecord(
            text=f"Sample text {i}",
            label=f"id_{i}",
            provenance=ProvenanceMetadata(record_index=i),
        )


def test_modulo_partitioning_success() -> None:
    """Test modulo assignment logic."""
    train_id = PartitionId(name="train")
    test_id = PartitionId(name="test")

    class DummyRecord(TaskRecord):
        int_field: int

    records = [
        DummyRecord(int_field=i, provenance=ProvenanceMetadata(record_index=i))
        for i in range(10)
    ]

    mapping = PartitionMapping(
        definition=ModuloPartitionDefinition(
            selector=SimpleFieldSelector(field_name="int_field"),
            modulo_divisor=10,
            partition_map={
                train_id: frozenset([0, 1, 2, 3, 4, 5, 6, 7]),
                test_id: frozenset([8, 9]),
            },
        ),
        strategy=ModuloPartitioner(),
    )

    partitioner = DatasetPartitioner()
    results = list(partitioner.partition(iter(records), mapping))

    assert len(results) == 10
    assert sum(1 for r in results if r.partition == train_id) == 8
    assert sum(1 for r in results if r.partition == test_id) == 2


def test_modulo_partitioning_invalid_type() -> None:
    """Test modulo partitioner raises error on non-integer fields."""
    mapping = PartitionMapping(
        definition=ModuloPartitionDefinition(
            selector=SimpleFieldSelector(field_name="text"),
            modulo_divisor=10,
            partition_map={PartitionId(name="all"): frozenset([0])},
        ),
        strategy=ModuloPartitioner(),
    )
    partitioner = DatasetPartitioner()

    with pytest.raises(PartitionAssignmentError, match="requires an integer field"):
        list(partitioner.partition(_mock_classification_stream(1), mapping))


def test_hash_partitioning_success() -> None:
    """Test deterministic hash assignment logic."""
    train_id = PartitionId(name="train")
    test_id = PartitionId(name="test")

    mapping = PartitionMapping(
        definition=HashPartitionDefinition(
            selector=SimpleFieldSelector(field_name="text"),
            hash_buckets=100,
            partition_map={
                train_id: frozenset(range(80)),
                test_id: frozenset(range(80, 100)),
            },
        ),
        strategy=HashPartitioner(),
    )

    partitioner = DatasetPartitioner()
    results = list(partitioner.partition(_mock_classification_stream(10), mapping))

    # Should be deterministic
    assert len(results) == 10
    assert all(isinstance(r, PartitionedRecord) for r in results)


def test_hash_partitioning_invalid_type() -> None:
    """Test hash partitioner raises error on non-string fields."""

    class DummyRecord(TaskRecord):
        int_field: int

    records = [DummyRecord(int_field=1, provenance=ProvenanceMetadata(record_index=1))]

    mapping = PartitionMapping(
        definition=HashPartitionDefinition(
            selector=SimpleFieldSelector(field_name="int_field"),
            hash_buckets=10,
            partition_map={PartitionId(name="all"): frozenset([0])},
        ),
        strategy=HashPartitioner(),
    )

    partitioner = DatasetPartitioner()
    with pytest.raises(PartitionAssignmentError, match="requires a string field"):
        list(partitioner.partition(iter(records), mapping))


def test_partition_mapping_validation() -> None:
    """Test that definition-to-strategy mismatch raises PartitionConfigurationError at instantiation."""
    with pytest.raises(
        PartitionConfigurationError,
        match="ModuloPartitioner explicitly requires a ModuloPartitionDefinition",
    ):
        PartitionMapping(
            definition=HashPartitionDefinition(
                selector=SimpleFieldSelector(field_name="text"),
                hash_buckets=10,
                partition_map={},
            ),
            strategy=ModuloPartitioner(),
        )


def test_partitioned_record_immutability() -> None:
    """Ensure output records maintain identity and immutability."""
    record = ClassificationRecord(
        text="text", label="label", provenance=ProvenanceMetadata(record_index=1)
    )
    p_record = PartitionedRecord(partition=PartitionId(name="train"), record=record)

    with pytest.raises(Exception):
        p_record.partition = PartitionId(name="test")  # type: ignore

    assert id(p_record.record) == id(record)


def test_implicit_drop() -> None:
    """Ensure records that don't match any partition map rule are silently dropped."""
    train_id = PartitionId(name="train")

    class DummyRecord(TaskRecord):
        int_field: int

    records = [
        DummyRecord(int_field=i, provenance=ProvenanceMetadata(record_index=i))
        for i in range(5)
    ]

    mapping = PartitionMapping(
        definition=ModuloPartitionDefinition(
            selector=SimpleFieldSelector(field_name="int_field"),
            modulo_divisor=10,
            # Only maps remainder 0, drops 1-9
            partition_map={train_id: frozenset([0])},
        ),
        strategy=ModuloPartitioner(),
    )

    partitioner = DatasetPartitioner()
    results = list(partitioner.partition(iter(records), mapping))

    assert len(results) == 1
    assert results[0].partition == train_id
    assert getattr(results[0].record, "int_field") == 0


def test_unexpected_exception_wrapping() -> None:
    """Test that unexpected exceptions are wrapped in PartitionExecutionError."""

    class FaultyPartitioner:
        def assign(
            self, record: TaskRecord, definition: typing.Any
        ) -> PartitionId | None:
            raise ValueError("Something catastrophic")

    mapping = PartitionMapping(
        definition=HashPartitionDefinition(
            selector=SimpleFieldSelector(field_name="text"),
            hash_buckets=10,
            partition_map={},
        ),
        strategy=FaultyPartitioner(),
    )

    partitioner = DatasetPartitioner()
    with pytest.raises(PartitionExecutionError, match="Partition execution failed"):
        list(partitioner.partition(_mock_classification_stream(1), mapping))
