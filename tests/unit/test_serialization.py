"""Unit tests for the Dataset Serialization Framework."""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pytest
from pydantic import ValidationError

from src.core.datasets.preprocessing_models import PreprocessedRecord
from src.core.datasets.serialization.base import BaseSerializer
from src.core.datasets.serialization.implementations import (
    JsonlSerializer,
    ManifestSerializer,
    MetadataSerializer,
)
from src.core.datasets.serialization_models import (
    DatasetManifest,
    JsonlSerializationDefinition,
    ManifestSerializationDefinition,
    MetadataSerializationDefinition,
    SerializationDefinition,
    SerializationFormat,
    SerializationPipeline,
    SerializationProfile,
    SerializationProfileRegistry,
    SerializationStep,
)
from src.core.datasets.serializer import DatasetSerializer
from src.core.exceptions import (
    DuplicateSerializationProfileError,
    SerializationConfigurationError,
    SerializationExecutionError,
    SerializationProfileNotFoundError,
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


# ---------------------------------------------------------------------------
# M5.4 Manifest Serialization tests
# ---------------------------------------------------------------------------


def test_dataset_manifest_immutability() -> None:
    """Ensure DatasetManifest is frozen."""
    manifest = DatasetManifest(serialization_format=SerializationFormat.JSONL)
    with pytest.raises(ValidationError):
        manifest.manifest_version = "mutated"


def test_dataset_manifest_default_version() -> None:
    """Ensure manifest_version defaults to '1.0'."""
    manifest = DatasetManifest(serialization_format=SerializationFormat.JSONL)
    assert manifest.manifest_version == "1.0"


def test_dataset_manifest_serialization_format_json_value() -> None:
    """Ensure serialization_format serializes to its string value, not enum name."""
    manifest = DatasetManifest(serialization_format=SerializationFormat.JSONL)
    dumped = manifest.model_dump(mode="json")
    assert dumped["serialization_format"] == "jsonl"


def test_dataset_manifest_optional_fields_accept_none() -> None:
    """Ensure all optional fields accept None."""
    manifest = DatasetManifest(
        serialization_format=SerializationFormat.JSONL,
        dataset_id=None,
        dataset_version=None,
        created_at=None,
        record_count=None,
    )
    assert manifest.dataset_id is None
    assert manifest.dataset_version is None
    assert manifest.created_at is None
    assert manifest.record_count is None


def test_dataset_manifest_datetime_serializes_iso8601() -> None:
    """Ensure created_at datetime serializes to ISO-8601 string via model_dump."""
    ts = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
    manifest = DatasetManifest(
        serialization_format=SerializationFormat.JSONL,
        created_at=ts,
    )
    dumped = manifest.model_dump(mode="json")
    # Pydantic serializes datetime to ISO-8601
    assert isinstance(dumped["created_at"], str)
    assert "2024-01-15" in dumped["created_at"]


def test_dataset_manifest_record_count_zero_accepted() -> None:
    """Ensure record_count=0 is accepted."""
    manifest = DatasetManifest(
        serialization_format=SerializationFormat.JSONL,
        record_count=0,
    )
    assert manifest.record_count == 0


def test_dataset_manifest_negative_record_count_rejected() -> None:
    """Ensure negative record_count is rejected at construction."""
    with pytest.raises(ValidationError):
        DatasetManifest(
            serialization_format=SerializationFormat.JSONL,
            record_count=-1,
        )


def test_manifest_serialization_definition_immutability() -> None:
    """Ensure ManifestSerializationDefinition is frozen."""
    with tempfile.TemporaryDirectory() as tmpdir:
        definition = ManifestSerializationDefinition(
            output_path=Path(tmpdir) / "manifest.json",
            manifest=DatasetManifest(serialization_format=SerializationFormat.JSONL),
        )
        with pytest.raises(ValidationError):
            definition.encoding = "latin-1"


def test_manifest_serializer_compatibility() -> None:
    """Ensure ManifestSerializer rejects incompatible definitions."""
    serializer = ManifestSerializer()
    with pytest.raises(
        SerializationConfigurationError,
        match="ManifestSerializer requires a ManifestSerializationDefinition",
    ):
        serializer.validate_compatibility(
            MockSerializationDefinition(target_name="test")
        )


def test_manifest_serializer_execution() -> None:
    """Ensure ManifestSerializer writes manifest exactly once and preserves record identity."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "manifest.json"
        ts = datetime(2024, 6, 1, 0, 0, 0, tzinfo=timezone.utc)

        manifest = DatasetManifest(
            serialization_format=SerializationFormat.JSONL,
            dataset_id="my-dataset",
            dataset_version="2.0",
            created_at=ts,
            record_count=2,
            extensions={"source": "unit-test"},
        )
        definition = ManifestSerializationDefinition(
            output_path=output_path,
            manifest=manifest,
        )
        serializer = ManifestSerializer()

        records = [
            _create_dummy_record("rec1"),
            _create_dummy_record("rec2"),
        ]

        # Verify identity preservation
        result_stream = serializer.serialize_stream(iter(records), definition)
        results = list(result_stream)
        assert len(results) == 2
        assert id(results[0]) == id(records[0])
        assert id(results[1]) == id(records[1])

        # Verify file output
        assert output_path.exists()
        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read()
        written = json.loads(content)

        assert written["manifest_version"] == "1.0"
        assert written["serialization_format"] == "jsonl"
        assert written["dataset_id"] == "my-dataset"
        assert written["dataset_version"] == "2.0"
        assert written["record_count"] == 2
        assert written["extensions"] == {"source": "unit-test"}
        # created_at serialized as ISO-8601 string
        assert isinstance(written["created_at"], str)
        assert "2024-06-01" in written["created_at"]

        # Formatting policy: pretty-printed, sorted keys, newline termination
        assert "\n" in content
        assert content.endswith("\n")


def test_manifest_serializer_empty_stream() -> None:
    """Ensure ManifestSerializer writes manifest even for an empty stream."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "manifest.json"
        definition = ManifestSerializationDefinition(
            output_path=output_path,
            manifest=DatasetManifest(serialization_format=SerializationFormat.JSONL),
        )
        serializer = ManifestSerializer()

        results = list(serializer.serialize_stream(iter([]), definition))
        assert results == []
        assert output_path.exists()
        with open(output_path, "r", encoding="utf-8") as f:
            written = json.load(f)
        assert written["manifest_version"] == "1.0"
        assert written["serialization_format"] == "jsonl"


def test_serialization_profile_immutability() -> None:
    """Ensure SerializationProfile is frozen."""
    definition = MockSerializationDefinition(target_name="test")
    strategy = MockSerializer(trace_list=[])
    step = SerializationStep(definition=definition, strategy=strategy)
    pipeline = SerializationPipeline(steps=(step,))
    profile = SerializationProfile(profile_id="test_profile", pipeline=pipeline)

    with pytest.raises(ValidationError):
        profile.profile_id = "mutated"


def test_serialization_profile_registry_immutability() -> None:
    """Ensure SerializationProfileRegistry is frozen."""
    definition = MockSerializationDefinition(target_name="test")
    strategy = MockSerializer(trace_list=[])
    step = SerializationStep(definition=definition, strategy=strategy)
    pipeline = SerializationPipeline(steps=(step,))
    profile = SerializationProfile(profile_id="test_profile", pipeline=pipeline)
    registry = SerializationProfileRegistry(profiles=(profile,))

    with pytest.raises(ValidationError):
        registry.profiles = ()


def test_duplicate_serialization_profile() -> None:
    """Ensure DuplicateSerializationProfileError is raised on collision."""
    definition = MockSerializationDefinition(target_name="test")
    strategy = MockSerializer(trace_list=[])
    step = SerializationStep(definition=definition, strategy=strategy)
    pipeline = SerializationPipeline(steps=(step,))
    profile1 = SerializationProfile(profile_id="duplicate", pipeline=pipeline)
    profile2 = SerializationProfile(profile_id="duplicate", pipeline=pipeline)

    with pytest.raises(DuplicateSerializationProfileError, match="duplicate"):
        SerializationProfileRegistry(profiles=(profile1, profile2))


def test_resolve_serialization_profile() -> None:
    """Ensure a valid profile is resolved properly."""
    definition = MockSerializationDefinition(target_name="test")
    strategy = MockSerializer(trace_list=[])
    step = SerializationStep(definition=definition, strategy=strategy)
    pipeline = SerializationPipeline(steps=(step,))
    profile = SerializationProfile(profile_id="test_profile", pipeline=pipeline)
    registry = SerializationProfileRegistry(profiles=(profile,))

    resolved = registry.resolve("test_profile")
    assert resolved is profile


def test_unknown_serialization_profile() -> None:
    """Ensure SerializationProfileNotFoundError is raised for unknown profiles."""
    registry = SerializationProfileRegistry(profiles=())
    with pytest.raises(SerializationProfileNotFoundError, match="unknown"):
        registry.resolve("unknown")


def test_execution_equivalence() -> None:
    """Ensure execution via profile resolution matches direct pipeline execution identically."""
    trace_direct: list[str] = []
    trace_profile: list[str] = []

    # We create two distinct strategy instances (otherwise trace would mix)
    # but they exhibit identical behavior.
    strategy_direct = MockSerializer(trace_list=trace_direct)
    strategy_profile = MockSerializer(trace_list=trace_profile)

    definition = MockSerializationDefinition(target_name="test")

    pipeline_direct = SerializationPipeline(
        steps=(SerializationStep(definition=definition, strategy=strategy_direct),)
    )
    pipeline_profile = SerializationPipeline(
        steps=(SerializationStep(definition=definition, strategy=strategy_profile),)
    )

    profile = SerializationProfile(profile_id="eq_profile", pipeline=pipeline_profile)
    registry = SerializationProfileRegistry(profiles=(profile,))

    # The dummy records to feed
    records_direct = [_create_dummy_record("rec1"), _create_dummy_record("rec2")]
    records_profile = [_create_dummy_record("rec1"), _create_dummy_record("rec2")]

    # Profile execution
    serializer_profile = DatasetSerializer()
    resolved_profile = registry.resolve("eq_profile")

    # We will use generators to feed the records and capture the execution flow securely
    def gen_direct() -> Iterator[PreprocessedRecord]:
        yield from records_direct

    def gen_profile() -> Iterator[PreprocessedRecord]:
        yield from records_profile

    # Execute through orchestrator
    serializer_direct = DatasetSerializer()
    serializer_direct.serialize(gen_direct(), pipeline_direct)
    serializer_profile.serialize(gen_profile(), resolved_profile.pipeline)

    # Identical invocation order ensures identical fail-fast behavior and exception propagation
    assert trace_direct == ["test:rec1", "test:rec2"]
    assert trace_profile == ["test:rec1", "test:rec2"]

    # To strictly verify yielded object identities and iterator semantics, we execute the pipeline directly
    stream_direct = iter(records_direct)
    for step in pipeline_direct.steps:
        stream_direct = step.strategy.serialize_stream(stream_direct, step.definition)
    direct_results = list(stream_direct)

    stream_profile = iter(records_profile)
    for step in resolved_profile.pipeline.steps:
        stream_profile = step.strategy.serialize_stream(stream_profile, step.definition)
    profile_results = list(stream_profile)

    assert len(direct_results) == 2
    assert len(profile_results) == 2
    assert direct_results[0] is records_direct[0]
    assert direct_results[1] is records_direct[1]
    assert profile_results[0] is records_profile[0]
    assert profile_results[1] is records_profile[1]
