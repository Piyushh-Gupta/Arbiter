"""Unit tests for the Schema Mapping Layer (M2.4)."""

from collections.abc import Iterator
from typing import Any

import pytest

from src.core.datasets.mapper import DatasetMapper
from src.core.datasets.mapping.base import TaskMapping
from src.core.datasets.mapping.implementations import (
    ClassificationMapper,
    FactVerificationMapper,
    QAMapper,
)
from src.core.datasets.mapping.registry import MappingRegistry
from src.core.datasets.mapping_models import (
    ClassificationRecord,
    FactVerificationRecord,
    QARecord,
    SchemaDefinition,
    TaskRecord,
    TaskSchemaType,
)
from src.core.datasets.normalization_models import NormalizedRecord, ProvenanceMetadata
from src.core.exceptions import (
    MissingRequiredFieldError,
    SchemaMappingError,
    UnsupportedTaskSchemaError,
)


def _mock_normalized_stream(
    count: int = 3, missing_field: bool = False, missing_dict: bool = False
) -> Iterator[NormalizedRecord]:
    for i in range(1, count + 1):
        provenance = ProvenanceMetadata(record_index=i)

        if missing_dict:
            # content is not a dict
            yield NormalizedRecord(content="just string", provenance=provenance)
            continue

        content: dict[str, Any] = {
            "src_claim": f"claim_{i}",
            "src_evidence": f"evidence_{i}",
            "src_label": f"label_{i}",
            "src_q": f"question_{i}",
            "src_c": f"context_{i}",
            "src_a": f"answer_{i}",
            "src_text": f"text_{i}",
        }

        if missing_field:
            # Delete required field to test MissingRequiredFieldError
            del content["src_claim"]

        yield NormalizedRecord(content=content, provenance=provenance)


@pytest.fixture
def fv_schema() -> SchemaDefinition:
    return SchemaDefinition(
        task_type=TaskSchemaType.FACT_VERIFICATION,
        field_map={
            "claim": "src_claim",
            "evidence": "src_evidence",
            "label": "src_label",
        },
    )


@pytest.fixture
def qa_schema() -> SchemaDefinition:
    return SchemaDefinition(
        task_type=TaskSchemaType.QA,
        field_map={
            "question": "src_q",
            "context": "src_c",
            "answer": "src_a",
        },
    )


@pytest.fixture
def cls_schema() -> SchemaDefinition:
    return SchemaDefinition(
        task_type=TaskSchemaType.CLASSIFICATION,
        field_map={
            "text": "src_text",
            "label": "src_label",
        },
    )


@pytest.fixture
def registry(fv_schema: SchemaDefinition) -> MappingRegistry:
    reg = MappingRegistry()
    reg.register(
        TaskSchemaType.FACT_VERIFICATION,
        TaskMapping(schema_definition=fv_schema, mapper=FactVerificationMapper()),
    )
    reg.freeze()
    return reg


@pytest.fixture
def mapper(registry: MappingRegistry) -> DatasetMapper:
    return DatasetMapper(registry=registry)


def test_registry_resolve(
    registry: MappingRegistry, fv_schema: SchemaDefinition
) -> None:
    """Test mapping registry resolves correct task mapping."""
    task_mapping = registry.resolve(TaskSchemaType.FACT_VERIFICATION)
    assert task_mapping.schema_definition == fv_schema
    assert isinstance(task_mapping.mapper, FactVerificationMapper)


def test_registry_unsupported_schema(registry: MappingRegistry) -> None:
    """Test registry raises UnsupportedTaskSchemaError for unknown tasks."""
    with pytest.raises(UnsupportedTaskSchemaError):
        registry.resolve(TaskSchemaType.QA)


def test_registry_freeze(registry: MappingRegistry) -> None:
    """Test registry cannot be modified after freezing."""
    with pytest.raises(
        RuntimeError, match="Cannot register mappings after initialization"
    ):
        registry.register(
            TaskSchemaType.QA,
            TaskMapping(
                schema_definition=SchemaDefinition(
                    task_type=TaskSchemaType.QA, field_map={}
                ),
                mapper=QAMapper(),
            ),
        )


def test_fact_verification_mapper(fv_schema: SchemaDefinition) -> None:
    """Test fact verification mapping strategy."""
    strategy = FactVerificationMapper()
    stream = _mock_normalized_stream(2)
    records = list(strategy.map_stream(stream, fv_schema))

    assert len(records) == 2
    assert all(isinstance(r, FactVerificationRecord) for r in records)
    assert records[0].claim == "claim_1"
    assert records[0].evidence == "evidence_1"
    assert records[0].label == "label_1"
    assert records[0].provenance.record_index == 1


def test_qa_mapper(qa_schema: SchemaDefinition) -> None:
    """Test QA mapping strategy."""
    strategy = QAMapper()
    stream = _mock_normalized_stream(1)
    records = list(strategy.map_stream(stream, qa_schema))

    assert len(records) == 1
    assert all(isinstance(r, QARecord) for r in records)
    assert records[0].question == "question_1"
    assert records[0].context == "context_1"
    assert records[0].answer == "answer_1"


def test_classification_mapper(cls_schema: SchemaDefinition) -> None:
    """Test Classification mapping strategy."""
    strategy = ClassificationMapper()
    stream = _mock_normalized_stream(1)
    records = list(strategy.map_stream(stream, cls_schema))

    assert len(records) == 1
    assert all(isinstance(r, ClassificationRecord) for r in records)
    assert records[0].text == "text_1"
    assert records[0].label == "label_1"


def test_mapper_orchestrator_success(mapper: DatasetMapper) -> None:
    """Test orchestrator maps successfully."""
    stream = _mock_normalized_stream(2)
    records = list(mapper.map(stream, TaskSchemaType.FACT_VERIFICATION))
    assert len(records) == 2


def test_mapper_missing_required_field(mapper: DatasetMapper) -> None:
    """Test MissingRequiredFieldError bubbles up natively."""
    stream = _mock_normalized_stream(1, missing_field=True)
    with pytest.raises(
        MissingRequiredFieldError, match="Required field 'src_claim' missing"
    ):
        list(mapper.map(stream, TaskSchemaType.FACT_VERIFICATION))


def test_mapper_missing_dict_content(mapper: DatasetMapper) -> None:
    """Test MissingRequiredFieldError when content is not a dict."""
    stream = _mock_normalized_stream(1, missing_dict=True)
    with pytest.raises(MissingRequiredFieldError, match="non-dictionary content"):
        list(mapper.map(stream, TaskSchemaType.FACT_VERIFICATION))


def test_mapper_schema_mapping_error(registry: MappingRegistry) -> None:
    """Test unknown exceptions are wrapped in SchemaMappingError."""

    class FaultyMapper:
        def map_stream(
            self, stream: Iterator[NormalizedRecord], schema_def: SchemaDefinition
        ) -> Iterator[TaskRecord]:
            raise ValueError("Something catastrophic")
            yield

    # Temporarily unfreeze to inject fault for test
    registry._frozen = False
    registry.register(
        TaskSchemaType.QA,
        TaskMapping(
            schema_definition=SchemaDefinition(
                task_type=TaskSchemaType.QA, field_map={"question": "q"}
            ),
            mapper=FaultyMapper(),
        ),
    )

    mapper = DatasetMapper(registry)
    stream = _mock_normalized_stream(1)

    with pytest.raises(SchemaMappingError, match="Schema mapping failed"):
        list(mapper.map(stream, TaskSchemaType.QA))
