"""Unit tests for the Dataset Preprocessing Layer (M3.1)."""

import typing
from collections.abc import Iterator

import pytest

from src.core.datasets.mapping_models import ClassificationRecord
from src.core.datasets.normalization_models import ProvenanceMetadata
from src.core.datasets.partitioning_models import PartitionedRecord, PartitionId
from src.core.datasets.preprocessing.base import (
    PreprocessingPipeline,
    PreprocessingStep,
)
from src.core.datasets.preprocessing.implementations import PassThroughPreprocessor
from src.core.datasets.preprocessing_models import (
    PassThroughPreprocessingDefinition,
    PreprocessedRecord,
    PreprocessingDefinition,
)
from src.core.datasets.preprocessor import DatasetPreprocessor
from src.core.exceptions import (
    PreprocessingConfigurationError,
    PreprocessingExecutionError,
)


def _mock_partitioned_stream(count: int = 5) -> Iterator[PartitionedRecord]:
    for i in range(1, count + 1):
        record = ClassificationRecord(
            text=f"Sample text {i}",
            label="SUPPORTS" if i % 2 == 0 else "REFUTES",
            provenance=ProvenanceMetadata(record_index=i),
        )
        yield PartitionedRecord(partition=PartitionId(name="train"), record=record)


def test_pass_through_preprocessing_success() -> None:
    """Test deterministic pass-through pipeline assignment logic."""
    pipeline = PreprocessingPipeline(
        steps=(
            PreprocessingStep(
                definition=PassThroughPreprocessingDefinition(),
                strategy=PassThroughPreprocessor(),
            ),
        )
    )

    preprocessor = DatasetPreprocessor()
    stream = _mock_partitioned_stream(5)
    records_list = list(stream)

    results = list(preprocessor.preprocess(iter(records_list), pipeline))

    assert len(results) == 5
    assert all(isinstance(r, PreprocessedRecord) for r in results)

    # Check identity and exact metadata structure
    for original, preprocessed in zip(records_list, results):
        assert id(original.record) == id(preprocessed.record)
        assert original.partition == preprocessed.partition
        assert hasattr(preprocessed, "preprocessing_metadata")


def test_preprocessing_configuration_validation() -> None:
    """Test that definition-to-strategy mismatch raises PreprocessingConfigurationError at instantiation."""

    class DummyDefinition:
        pass

    with pytest.raises(
        PreprocessingConfigurationError,
        match="PassThroughPreprocessor requires a PassThroughPreprocessingDefinition",
    ):
        PreprocessingStep(
            definition=typing.cast(PreprocessingDefinition, DummyDefinition()),
            strategy=PassThroughPreprocessor(),
        )


def test_multiple_pass_through_pipeline() -> None:
    """Ensure pipeline handles multiple passes seamlessly via generators."""
    pipeline = PreprocessingPipeline(
        steps=(
            PreprocessingStep(
                definition=PassThroughPreprocessingDefinition(),
                strategy=PassThroughPreprocessor(),
            ),
            PreprocessingStep(
                definition=PassThroughPreprocessingDefinition(),
                strategy=PassThroughPreprocessor(),
            ),
            PreprocessingStep(
                definition=PassThroughPreprocessingDefinition(),
                strategy=PassThroughPreprocessor(),
            ),
        )
    )

    preprocessor = DatasetPreprocessor()
    results = list(preprocessor.preprocess(_mock_partitioned_stream(10), pipeline))

    assert len(results) == 10
    assert all(isinstance(r, PreprocessedRecord) for r in results)


def test_unexpected_exception_wrapping() -> None:
    """Test that unexpected exceptions are wrapped in PreprocessingExecutionError."""

    class FaultyPreprocessor:
        def process_stream(
            self,
            stream: Iterator[PreprocessedRecord],
            definition: PreprocessingDefinition,
        ) -> Iterator[PreprocessedRecord]:
            raise ValueError("Something catastrophic")
            yield

        def validate_compatibility(self, definition: PreprocessingDefinition) -> None:
            pass

    pipeline = PreprocessingPipeline(
        steps=(
            PreprocessingStep(
                definition=PassThroughPreprocessingDefinition(),
                strategy=FaultyPreprocessor(),
            ),
        )
    )

    preprocessor = DatasetPreprocessor()
    with pytest.raises(PreprocessingExecutionError, match="Pipeline execution failed"):
        list(preprocessor.preprocess(_mock_partitioned_stream(1), pipeline))
