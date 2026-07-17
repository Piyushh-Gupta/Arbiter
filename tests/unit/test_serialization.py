"""Unit tests for the Dataset Serialization Framework."""

import json
import tempfile
from pathlib import Path
from typing import Iterator

import pytest
from pydantic import ValidationError

from src.core.datasets.preprocessing_models import PreprocessedRecord
from src.core.datasets.serialization.base import BaseSerializer
from src.core.datasets.serialization.implementations import (
    JsonlSerializer,
    MetadataSerializer,
)
from src.core.datasets.serialization_models import (
    JsonlSerializationDefinition,
    MetadataSerializationDefinition,
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


def test_jsonl_serializer_compatibility() -> None:
    """Ensure JsonlSerializer rejects invalid definitions."""
    serializer = JsonlSerializer()

    with pytest.raises(
        SerializationConfigurationError,
        match="JsonlSerializer requires a JsonlSerializationDefinition",
    ):
        serializer.validate_compatibility(
            MockSerializationDefinition(target_name="test")
        )


def test_jsonl_serializer_execution() -> None:
    """Ensure JsonlSerializer writes valid JSONL output deterministically without buffering."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "output.jsonl"

        definition = JsonlSerializationDefinition(output_path=output_path)
        serializer = JsonlSerializer()

        step = SerializationStep(definition=definition, strategy=serializer)
        pipeline = SerializationPipeline(steps=(step,))

        records = [
            _create_dummy_record("rec1"),
            _create_dummy_record("rec2"),
        ]

        orchestrator = DatasetSerializer()

        # Act
        # Notice we are passing a generator to ensure stream isolation
        def record_generator() -> Iterator[PreprocessedRecord]:
            yield from records

        orchestrator.serialize(record_generator(), pipeline)

        # Verify Identity Preservation & Memory (Execution equivalence)
        # We manually verify by pulling from the serializer stream directly
        result_stream = serializer.serialize_stream(iter(records), definition)
        results = list(result_stream)
        assert len(results) == 2
        assert id(results[0]) == id(records[0])
        assert id(results[1]) == id(records[1])

        # Verify File Output
        assert output_path.exists()

        with open(output_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) == 2

        # Verify newline delimited and parseable
        doc1 = json.loads(lines[0])
        doc2 = json.loads(lines[1])

        assert doc1["text"] == "test_text"
        # The index we assign in `_create_dummy_record` based on task_id
        assert doc1["provenance"]["record_index"] == 1
        assert doc2["provenance"]["record_index"] == 2


def test_metadata_serializer_compatibility() -> None:
    """Ensure MetadataSerializer rejects invalid definitions."""
    serializer = MetadataSerializer()

    with pytest.raises(
        SerializationConfigurationError,
        match="MetadataSerializer requires a MetadataSerializationDefinition",
    ):
        serializer.validate_compatibility(
            MockSerializationDefinition(target_name="test")
        )


def test_metadata_serializer_execution() -> None:
    """Ensure MetadataSerializer writes metadata exactly once and preserves identity."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "metadata.json"
        from typing import Any

        metadata_map: dict[str, Any] = {
            "dataset": "test",
            "version": "1.0",
            "stats": {"count": 2},
        }

        definition = MetadataSerializationDefinition(
            output_path=output_path, metadata=metadata_map
        )
        serializer = MetadataSerializer()

        step = SerializationStep(definition=definition, strategy=serializer)
        pipeline = SerializationPipeline(steps=(step,))

        records = [
            _create_dummy_record("rec1"),
            _create_dummy_record("rec2"),
        ]

        orchestrator = DatasetSerializer()

        def record_generator() -> Iterator[PreprocessedRecord]:
            yield from records

        orchestrator.serialize(record_generator(), pipeline)

        # Verify Identity Preservation & Memory (Execution equivalence)
        result_stream = serializer.serialize_stream(iter(records), definition)
        results = list(result_stream)
        assert len(results) == 2
        assert id(results[0]) == id(records[0])
        assert id(results[1]) == id(records[1])

        # Verify File Output
        assert output_path.exists()

        with open(output_path, "r", encoding="utf-8") as f:
            written_json = json.load(f)

        assert written_json == metadata_map

        # Verify formatting policy (indentation and newline termination)
        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Should be pretty printed (contains newlines inside json)
        assert "\n" in content
        # Ensure proper newline at the end
        assert content.endswith("\n")

        # Ensure it works for an empty stream without failing to write the metadata
        empty_output_path = Path(tmpdir) / "empty_metadata.json"
        empty_metadata_map: dict[str, Any] = {"empty": True}
        empty_definition = MetadataSerializationDefinition(
            output_path=empty_output_path, metadata=empty_metadata_map
        )
        empty_serializer = MetadataSerializer()
        empty_stream = empty_serializer.serialize_stream(iter([]), empty_definition)

        # Eagerly writing? We need to pull the first item or let it finish
        # Actually in python a generator doesn't execute anything until `next()` is called or iterated.
        # So we MUST iterate it.
        list(empty_stream)
        assert empty_output_path.exists()
        with open(empty_output_path, "r", encoding="utf-8") as f:
            empty_json = json.load(f)
        assert empty_json == {"empty": True}
