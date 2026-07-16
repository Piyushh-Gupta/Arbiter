"""Unit tests for the M2.3 Dataset Normalization Layer."""

from collections.abc import Iterator

import pytest
from pydantic import ValidationError

from src.core.datasets.normalization_models import NormalizedRecord
from src.core.datasets.normalizer import DatasetNormalizer
from src.core.datasets.parser_models import ParsedRecord
from src.core.exceptions import (
    MalformedNormalizedRecordError,
    NormalizationFailureError,
)


@pytest.fixture
def normalizer() -> DatasetNormalizer:
    return DatasetNormalizer()


def _mock_stream(count: int, invalid: bool = False) -> Iterator[ParsedRecord]:
    for i in range(count):
        if invalid:
            # We bypass type checking and pydantic validation at runtime deliberately to test failure
            yield ParsedRecord.model_construct(data=set())
        else:
            yield ParsedRecord(data={"id": i, "nested": {"val": f"test_{i}"}})


def test_normalization_1_to_1_guarantee(normalizer: DatasetNormalizer) -> None:
    """Test the pipeline correctly yields 1-to-1 canonical structures."""
    count = 5
    stream = _mock_stream(count)

    results = list(normalizer.normalize(stream))
    assert len(results) == count

    for idx, record in enumerate(results, start=1):
        assert isinstance(record, NormalizedRecord)
        assert record.provenance.record_index == idx
        # Verify content preservation
        content = record.content
        assert isinstance(content, dict)
        assert content["id"] == idx - 1
        assert content["nested"] == {"val": f"test_{idx - 1}"}


def test_normalization_immutability(normalizer: DatasetNormalizer) -> None:
    """Test the pipeline produces strictly immutable structures."""
    stream = _mock_stream(1)
    results = list(normalizer.normalize(stream))
    record = results[0]

    with pytest.raises(ValidationError):
        record.provenance.record_index = 999

    with pytest.raises(ValidationError):
        record.content = {"new": "data"}


def test_normalization_string_payload(normalizer: DatasetNormalizer) -> None:
    """Test the pipeline supports plain text strings instead of dictionaries."""

    def string_stream() -> Iterator[ParsedRecord]:
        yield ParsedRecord(data="raw string line")

    results = list(normalizer.normalize(string_stream()))
    assert len(results) == 1
    assert results[0].content == "raw string line"


def test_normalization_malformed_record(normalizer: DatasetNormalizer) -> None:
    """Test validation failure is correctly caught and wrapped."""
    stream = _mock_stream(1, invalid=True)

    with pytest.raises(
        MalformedNormalizedRecordError,
        match="Failed to structural validation for record 1",
    ):
        list(normalizer.normalize(stream))


def test_normalization_internal_stream_failure(normalizer: DatasetNormalizer) -> None:
    """Test unexpected upstream iterator failures are correctly wrapped."""

    def failing_stream() -> Iterator[ParsedRecord]:
        yield ParsedRecord(data={"id": 1})
        raise RuntimeError("Stream broken")

    iterator = normalizer.normalize(failing_stream())

    # First record should yield correctly
    record = next(iterator)
    assert record.provenance.record_index == 1

    # Second evaluation should raise domain failure
    with pytest.raises(
        NormalizationFailureError, match="Normalization pipeline failed"
    ):
        next(iterator)
