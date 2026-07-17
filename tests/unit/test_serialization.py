"""Unit tests for the Dataset Serialization Framework."""

from typing import Iterator

import pytest
from pydantic import ValidationError

from src.core.datasets.preprocessing_models import PreprocessedRecord
from src.core.datasets.serialization.base import BaseSerializer
from src.core.datasets.serialization_models import (
    SerializationDefinition,
    SerializationPipeline,
    SerializationStep,
)
from src.core.datasets.serializer import DatasetSerializer
from src.core.exceptions import (
    SerializationConfigurationError,
    SerializationExecutionError,
)


class MockSerializationDefinition(SerializationDefinition):
    """A mock configuration for testing."""

    target_name: str


class MockSerializer(BaseSerializer):
    """A mock strategy that appends the target name to a global list and yields exactly identical records."""

    def __init__(self, trace_list: list[str]) -> None:
        self.trace_list = trace_list

    def serialize_stream(
        self,
        stream: Iterator[PreprocessedRecord],
        definition: SerializationDefinition,
    ) -> Iterator[PreprocessedRecord]:
        assert isinstance(definition, MockSerializationDefinition)

        for record in stream:
            self.trace_list.append(
                f"{definition.target_name}:rec{record.record.provenance.record_index}"
            )
            # Strict object identity preservation
            yield record

    def validate_compatibility(self, definition: SerializationDefinition) -> None:
        if not isinstance(definition, MockSerializationDefinition):
            raise SerializationConfigurationError("Incompatible definition type.")


class BadMockSerializer(BaseSerializer):
    """A mock strategy that always fails compatibility validation."""

    def serialize_stream(
        self,
        stream: Iterator[PreprocessedRecord],
        definition: SerializationDefinition,
    ) -> Iterator[PreprocessedRecord]:
        yield from stream

    def validate_compatibility(self, definition: SerializationDefinition) -> None:
        raise SerializationConfigurationError("Always fails.")


class ErrorMockSerializer(BaseSerializer):
    """A mock strategy that raises an execution error mid-stream."""

    def serialize_stream(
        self,
        stream: Iterator[PreprocessedRecord],
        definition: SerializationDefinition,
    ) -> Iterator[PreprocessedRecord]:
        for idx, record in enumerate(stream):
            if idx == 1:
                raise SerializationExecutionError("Failed to write to disk.")
            yield record

    def validate_compatibility(self, definition: SerializationDefinition) -> None:
        pass


def _create_dummy_record(task_id: str) -> PreprocessedRecord:
    from src.core.datasets.mapping_models import ClassificationRecord
    from src.core.datasets.normalization_models import ProvenanceMetadata
    from src.core.datasets.partitioning_models import PartitionId
    from src.core.datasets.preprocessing_models import (
        PreprocessedRecord,
        PreprocessingMetadata,
    )

    return PreprocessedRecord(
        record=ClassificationRecord(
            provenance=ProvenanceMetadata(record_index=int(task_id.replace("rec", ""))),
            text="test_text",
            label=None,
        ),
        partition=PartitionId(name="train"),
        preprocessing_metadata=PreprocessingMetadata(),
    )


def test_serialization_configuration() -> None:
    """Ensure SerializationStep validates compatibility strictly during construction."""
    # Pass compatible
    definition = MockSerializationDefinition(target_name="jsonl")
    strategy = MockSerializer(trace_list=[])
    step = SerializationStep(definition=definition, strategy=strategy)

    assert step.definition is definition
    assert step.strategy is strategy

    # Fail incompatible natively
    with pytest.raises(SerializationConfigurationError, match="Always fails"):
        SerializationStep(definition=definition, strategy=BadMockSerializer())

    # Check Pipeline Construction
    pipeline = SerializationPipeline(steps=(step,))
    assert len(pipeline.steps) == 1

    # Missing steps raises error (no default empty tuple)
    with pytest.raises(ValidationError):
        SerializationPipeline()  # type: ignore[call-arg]


def test_serialization_execution_flow() -> None:
    """Ensure DatasetSerializer drives the iterator exactly without buffering."""
    trace: list[str] = []

    step1 = SerializationStep(
        definition=MockSerializationDefinition(target_name="A"),
        strategy=MockSerializer(trace),
    )
    step2 = SerializationStep(
        definition=MockSerializationDefinition(target_name="B"),
        strategy=MockSerializer(trace),
    )

    pipeline = SerializationPipeline(steps=(step1, step2))

    # Notice we use a generator here to verify stream is consumed incrementally cleanly
    def record_generator() -> Iterator[PreprocessedRecord]:
        trace.append("gen:yield1")
        yield _create_dummy_record("rec1")
        trace.append("gen:yield2")
        yield _create_dummy_record("rec2")

    serializer = DatasetSerializer()

    # Act
    serializer.serialize(record_generator(), pipeline)

    # Verify execution order (generator interleaved natively with serializers)
    assert trace == ["gen:yield1", "A:rec1", "B:rec1", "gen:yield2", "A:rec2", "B:rec2"]


def test_serialization_fail_fast() -> None:
    """Ensure serialization exceptions halt the stream immediately cleanly safely."""
    trace: list[str] = []

    step1 = SerializationStep(
        definition=MockSerializationDefinition(target_name="A"),
        strategy=MockSerializer(trace),
    )
    step2 = SerializationStep(
        definition=MockSerializationDefinition(target_name="Error"),
        strategy=ErrorMockSerializer(),
    )
    step3 = SerializationStep(
        definition=MockSerializationDefinition(target_name="B"),
        strategy=MockSerializer(trace),
    )

    pipeline = SerializationPipeline(steps=(step1, step2, step3))

    def record_generator() -> Iterator[PreprocessedRecord]:
        trace.append("gen:yield1")
        yield _create_dummy_record("rec1")
        trace.append("gen:yield2")
        yield _create_dummy_record("rec2")
        trace.append("gen:yield3")
        yield _create_dummy_record("rec3")

    serializer = DatasetSerializer()

    with pytest.raises(SerializationExecutionError, match="Failed to write to disk."):
        serializer.serialize(record_generator(), pipeline)

    # Verify stream halted precisely exactly cleanly completely
    assert trace == [
        "gen:yield1",
        "A:rec1",
        "B:rec1",
        "gen:yield2",
        "A:rec2",
        # Error occurs during step2 for rec2, halting execution before step3 ("B") and gen:yield3
    ]
