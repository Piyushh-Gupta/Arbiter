"""Unit tests for the Dataset Filtering and Selection Layer (M2.5)."""

import typing
from collections.abc import Iterator

import pytest
from pydantic import ValidationError

from src.core.datasets.filter import DatasetFilter
from src.core.datasets.filtering.implementations import (
    FieldEqualsFilter,
    FieldExistsFilter,
    FieldInSetFilter,
    FieldLengthFilter,
)
from src.core.datasets.filtering_models import (
    FieldEqualsPredicate,
    FieldExistsPredicate,
    FieldInSetPredicate,
    FieldLengthPredicate,
    FilterPipeline,
    FilterStep,
)
from src.core.datasets.mapping_models import ClassificationRecord, TaskRecord
from src.core.datasets.normalization_models import ProvenanceMetadata
from src.core.datasets.selectors import SimpleFieldSelector
from src.core.exceptions import FieldResolutionError, FilterExecutionError


def _mock_classification_stream(count: int = 5) -> Iterator[TaskRecord]:
    for i in range(1, count + 1):
        yield ClassificationRecord(
            text=f"Sample text {i}",
            label="SUPPORTS" if i % 2 == 0 else "REFUTES",
            provenance=ProvenanceMetadata(record_index=i),
        )


def test_simple_field_selector_success() -> None:
    record = ClassificationRecord(
        text="text", label="SUPPORTS", provenance=ProvenanceMetadata(record_index=1)
    )
    selector = SimpleFieldSelector(field_name="label")
    assert selector.resolve(record) == "SUPPORTS"


def test_simple_field_selector_failure() -> None:
    record = ClassificationRecord(
        text="text", label="SUPPORTS", provenance=ProvenanceMetadata(record_index=1)
    )
    selector = SimpleFieldSelector(field_name="nonexistent_field")
    with pytest.raises(FieldResolutionError, match="not found on record"):
        selector.resolve(record)


def test_field_exists_filter() -> None:
    """Test field exists filter."""

    def mock_stream() -> Iterator[TaskRecord]:
        yield ClassificationRecord(
            text="1", label="L1", provenance=ProvenanceMetadata(record_index=1)
        )
        yield ClassificationRecord(
            text="2", label=None, provenance=ProvenanceMetadata(record_index=2)
        )
        yield ClassificationRecord(
            text="3", label="L3", provenance=ProvenanceMetadata(record_index=3)
        )

    predicate = FieldExistsPredicate(selector=SimpleFieldSelector(field_name="label"))
    filter_strat = FieldExistsFilter()

    records = list(filter_strat.filter_stream(mock_stream(), predicate))
    assert len(records) == 2
    assert getattr(records[0], "text") == "1"
    assert getattr(records[1], "text") == "3"


def test_field_equals_filter() -> None:
    """Test field equals filter."""
    predicate = FieldEqualsPredicate(
        selector=SimpleFieldSelector(field_name="label"), target_value="SUPPORTS"
    )
    filter_strat = FieldEqualsFilter()

    records = list(
        filter_strat.filter_stream(_mock_classification_stream(4), predicate)
    )
    assert len(records) == 2
    assert all(getattr(r, "label") == "SUPPORTS" for r in records)


def test_field_in_set_filter() -> None:
    """Test field in set filter."""
    predicate = FieldInSetPredicate(
        selector=SimpleFieldSelector(field_name="label"),
        allowed_values=frozenset(["REFUTES"]),
    )
    filter_strat = FieldInSetFilter()

    records = list(
        filter_strat.filter_stream(_mock_classification_stream(4), predicate)
    )
    assert len(records) == 2
    assert all(getattr(r, "label") == "REFUTES" for r in records)


def test_field_length_filter() -> None:
    """Test field length filter."""

    def mock_stream() -> Iterator[TaskRecord]:
        yield ClassificationRecord(
            text="short", label="L1", provenance=ProvenanceMetadata(record_index=1)
        )
        yield ClassificationRecord(
            text="a bit longer",
            label="L2",
            provenance=ProvenanceMetadata(record_index=2),
        )
        yield ClassificationRecord(
            text="extremely long text string",
            label="L3",
            provenance=ProvenanceMetadata(record_index=3),
        )

    predicate = FieldLengthPredicate(
        selector=SimpleFieldSelector(field_name="text"), min_length=10, max_length=20
    )
    filter_strat = FieldLengthFilter()

    records = list(filter_strat.filter_stream(mock_stream(), predicate))
    assert len(records) == 1
    assert getattr(records[0], "text") == "a bit longer"


def test_filter_pipeline_identity_preservation() -> None:
    """Ensure accepted records are yielded unchanged and maintain identity."""
    stream = _mock_classification_stream(2)
    original_records = list(stream)

    pipeline = FilterPipeline(
        steps=(
            FilterStep(
                definition=FieldExistsPredicate(
                    selector=SimpleFieldSelector(field_name="label")
                ),
                filter_strategy=FieldExistsFilter(),
            ),
        )
    )

    dataset_filter = DatasetFilter()
    filtered_records = list(dataset_filter.filter(iter(original_records), pipeline))

    assert len(filtered_records) == 2
    assert id(filtered_records[0]) == id(original_records[0])
    assert id(filtered_records[1]) == id(original_records[1])


def test_filter_pipeline_composition() -> None:
    """Test combining multiple steps in a pipeline."""
    pipeline = FilterPipeline(
        steps=(
            FilterStep(
                definition=FieldExistsPredicate(
                    selector=SimpleFieldSelector(field_name="label")
                ),
                filter_strategy=FieldExistsFilter(),
            ),
            FilterStep(
                definition=FieldEqualsPredicate(
                    selector=SimpleFieldSelector(field_name="label"),
                    target_value="SUPPORTS",
                ),
                filter_strategy=FieldEqualsFilter(),
            ),
        )
    )

    dataset_filter = DatasetFilter()
    records = list(dataset_filter.filter(_mock_classification_stream(4), pipeline))

    assert len(records) == 2
    assert all(getattr(r, "label") == "SUPPORTS" for r in records)


def test_dataset_filter_exception_propagation() -> None:
    """Test that FieldResolutionError bubbles up natively."""
    pipeline = FilterPipeline(
        steps=(
            FilterStep(
                definition=FieldExistsPredicate(
                    selector=SimpleFieldSelector(field_name="missing_field")
                ),
                filter_strategy=FieldExistsFilter(),
            ),
        )
    )

    dataset_filter = DatasetFilter()

    with pytest.raises(FieldResolutionError, match="not found on record"):
        list(dataset_filter.filter(_mock_classification_stream(1), pipeline))


def test_dataset_filter_unexpected_exception() -> None:
    """Test that unexpected exceptions are wrapped in FilterExecutionError."""

    class FaultyFilter:
        def filter_stream(
            self, stream: Iterator[TaskRecord], definition: typing.Any
        ) -> Iterator[TaskRecord]:
            raise ValueError("Something catastrophic")
            yield

    pipeline = FilterPipeline(
        steps=(
            FilterStep(
                definition=FieldExistsPredicate(
                    selector=SimpleFieldSelector(field_name="label")
                ),
                filter_strategy=FaultyFilter(),
            ),
        )
    )

    dataset_filter = DatasetFilter()

    with pytest.raises(FilterExecutionError, match="Filter execution failed"):
        list(dataset_filter.filter(_mock_classification_stream(1), pipeline))


def test_task_record_immutability() -> None:
    """Ensure TaskRecords cannot be mutated during filtering."""
    record = ClassificationRecord(
        text="text", label="label", provenance=ProvenanceMetadata(record_index=1)
    )
    with pytest.raises(ValidationError):
        record.text = "new text"
