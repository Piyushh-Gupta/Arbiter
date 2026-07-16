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
from src.core.datasets.preprocessing.text_implementations import (
    ControlCharacterRemovalPreprocessor,
    UnicodeNormalizationPreprocessor,
    WhitespaceNormalizationPreprocessor,
)
from src.core.datasets.preprocessing.text_models import (
    ControlCharacterRemovalDefinition,
    UnicodeNormalizationDefinition,
    WhitespaceNormalizationDefinition,
)
from src.core.datasets.preprocessing_models import (
    PassThroughPreprocessingDefinition,
    PreprocessedRecord,
    PreprocessingDefinition,
)
from src.core.datasets.preprocessor import DatasetPreprocessor
from src.core.datasets.selectors import SimpleFieldSelector
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


def test_whitespace_normalization_collapsing_and_trimming() -> None:
    """Validate whitespace collapsing and edge trimming."""
    pipeline = PreprocessingPipeline(
        steps=(
            PreprocessingStep(
                definition=WhitespaceNormalizationDefinition(
                    selectors=(SimpleFieldSelector(field_name="text"),),
                    collapse_multiple=True,
                    trim_leading=True,
                    trim_trailing=True,
                ),
                strategy=WhitespaceNormalizationPreprocessor(),
            ),
        )
    )

    preprocessor = DatasetPreprocessor()

    # Custom stream with weird whitespace
    def _weird_whitespace_stream() -> Iterator[PartitionedRecord]:
        record = ClassificationRecord(
            text="   Hello \t\n  World   ",
            label="SUPPORTS",
            provenance=ProvenanceMetadata(record_index=1),
        )
        yield PartitionedRecord(partition=PartitionId(name="train"), record=record)

    results = list(preprocessor.preprocess(_weird_whitespace_stream(), pipeline))
    assert len(results) == 1
    processed = results[0].record
    assert isinstance(processed, ClassificationRecord)
    assert processed.text == "Hello World"
    # Ensure label was untouched
    assert processed.label == "SUPPORTS"


def test_whitespace_normalization_immutable_replacement() -> None:
    """Assert that transformations yield a distinctly new TaskRecord memory instance."""
    pipeline = PreprocessingPipeline(
        steps=(
            PreprocessingStep(
                definition=WhitespaceNormalizationDefinition(
                    selectors=(SimpleFieldSelector(field_name="text"),)
                ),
                strategy=WhitespaceNormalizationPreprocessor(),
            ),
        )
    )

    preprocessor = DatasetPreprocessor()
    stream = _mock_partitioned_stream(1)
    original_records = list(stream)

    results = list(preprocessor.preprocess(iter(original_records), pipeline))
    assert len(results) == 1

    original_record = original_records[0].record
    processed_record = results[0].record

    # TaskRecord identity is no longer preserved
    assert id(original_record) != id(processed_record)

    # But PartitionId and Provenance remain exactly the same
    assert original_records[0].partition == results[0].partition
    assert original_record.provenance == processed_record.provenance


def test_whitespace_normalization_configuration_validation() -> None:
    with pytest.raises(
        PreprocessingConfigurationError,
        match="WhitespaceNormalizationPreprocessor requires a WhitespaceNormalizationDefinition.",
    ):
        PreprocessingStep(
            definition=PassThroughPreprocessingDefinition(),
            strategy=WhitespaceNormalizationPreprocessor(),
        )


def test_whitespace_normalization_unicode_and_empty() -> None:
    """Assert non-whitespace unicode configurations seamlessly pass unharmed, empty strings handle well."""
    pipeline = PreprocessingPipeline(
        steps=(
            PreprocessingStep(
                definition=WhitespaceNormalizationDefinition(
                    selectors=(SimpleFieldSelector(field_name="text"),)
                ),
                strategy=WhitespaceNormalizationPreprocessor(),
            ),
        )
    )

    def _edge_case_stream() -> Iterator[PartitionedRecord]:
        for text in ["", "     ", "こんにちは世界", " already normal "]:
            record = ClassificationRecord(
                text=text,
                label="SUPPORTS",
                provenance=ProvenanceMetadata(record_index=1),
            )
            yield PartitionedRecord(partition=PartitionId(name="train"), record=record)

    results = list(DatasetPreprocessor().preprocess(_edge_case_stream(), pipeline))
    r0 = typing.cast(ClassificationRecord, results[0].record)
    r1 = typing.cast(ClassificationRecord, results[1].record)
    r2 = typing.cast(ClassificationRecord, results[2].record)
    r3 = typing.cast(ClassificationRecord, results[3].record)

    assert r0.text == ""
    assert r1.text == ""
    assert r2.text == "こんにちは世界"
    assert r3.text == "already normal"


def test_unicode_normalization_nfc_nfd() -> None:
    """Validate NFC and NFD composition/decomposition mappings correctly."""

    # NFD: Decomposed
    nfd_pipeline = PreprocessingPipeline(
        steps=(
            PreprocessingStep(
                definition=UnicodeNormalizationDefinition(
                    selectors=(SimpleFieldSelector(field_name="text"),),
                    normalization_form="NFD",
                ),
                strategy=UnicodeNormalizationPreprocessor(),
            ),
        )
    )

    # NFC: Composed
    nfc_pipeline = PreprocessingPipeline(
        steps=(
            PreprocessingStep(
                definition=UnicodeNormalizationDefinition(
                    selectors=(SimpleFieldSelector(field_name="text"),),
                    normalization_form="NFC",
                ),
                strategy=UnicodeNormalizationPreprocessor(),
            ),
        )
    )

    def _composed_stream() -> Iterator[PartitionedRecord]:
        record = ClassificationRecord(
            text="é",  # composed
            label="SUPPORTS",
            provenance=ProvenanceMetadata(record_index=1),
        )
        yield PartitionedRecord(partition=PartitionId(name="train"), record=record)

    def _decomposed_stream() -> Iterator[PartitionedRecord]:
        record = ClassificationRecord(
            text="e\u0301",  # decomposed
            label="SUPPORTS",
            provenance=ProvenanceMetadata(record_index=1),
        )
        yield PartitionedRecord(partition=PartitionId(name="train"), record=record)

    # Test NFD decomposing a composed character
    res_nfd = list(DatasetPreprocessor().preprocess(_composed_stream(), nfd_pipeline))
    r_nfd = typing.cast(ClassificationRecord, res_nfd[0].record)
    assert r_nfd.text == "e\u0301"

    # Test NFC composing a decomposed character
    res_nfc = list(DatasetPreprocessor().preprocess(_decomposed_stream(), nfc_pipeline))
    r_nfc = typing.cast(ClassificationRecord, res_nfc[0].record)
    assert r_nfc.text == "é"


def test_unicode_normalization_nfkc_nfkd() -> None:
    """Validate NFKC and NFKD compatibility mappings."""
    nfkc_pipeline = PreprocessingPipeline(
        steps=(
            PreprocessingStep(
                definition=UnicodeNormalizationDefinition(
                    selectors=(SimpleFieldSelector(field_name="text"),),
                    normalization_form="NFKC",
                ),
                strategy=UnicodeNormalizationPreprocessor(),
            ),
        )
    )

    def _compatibility_stream() -> Iterator[PartitionedRecord]:
        record = ClassificationRecord(
            text="ﬁ",  # ligature fi
            label="SUPPORTS",
            provenance=ProvenanceMetadata(record_index=1),
        )
        yield PartitionedRecord(partition=PartitionId(name="train"), record=record)

    res_nfkc = list(
        DatasetPreprocessor().preprocess(_compatibility_stream(), nfkc_pipeline)
    )
    r_nfkc = typing.cast(ClassificationRecord, res_nfkc[0].record)
    assert r_nfkc.text == "fi"


def test_unicode_normalization_preserves_non_selected() -> None:
    """Verify non-selected fields and provenance are preserved, empty works."""
    pipeline = PreprocessingPipeline(
        steps=(
            PreprocessingStep(
                definition=UnicodeNormalizationDefinition(
                    selectors=(SimpleFieldSelector(field_name="text"),),
                    normalization_form="NFC",
                ),
                strategy=UnicodeNormalizationPreprocessor(),
            ),
        )
    )

    def _stream() -> Iterator[PartitionedRecord]:
        record = ClassificationRecord(
            text="",  # empty
            label="é",  # composed, but NOT selected
            provenance=ProvenanceMetadata(record_index=1),
        )
        yield PartitionedRecord(partition=PartitionId(name="train"), record=record)

    res = list(DatasetPreprocessor().preprocess(_stream(), pipeline))
    r = typing.cast(ClassificationRecord, res[0].record)

    assert r.text == ""
    assert r.label == "é"
    assert r.provenance.record_index == 1


def test_unicode_normalization_configuration_validation() -> None:
    with pytest.raises(
        PreprocessingConfigurationError,
        match="UnicodeNormalizationPreprocessor requires a UnicodeNormalizationDefinition.",
    ):
        PreprocessingStep(
            definition=PassThroughPreprocessingDefinition(),
            strategy=UnicodeNormalizationPreprocessor(),
        )


def test_control_character_removal() -> None:
    """Test removal of ASCII and Unicode control characters with explicit bypassing."""
    pipeline = PreprocessingPipeline(
        steps=(
            PreprocessingStep(
                definition=ControlCharacterRemovalDefinition(
                    selectors=(SimpleFieldSelector(field_name="text"),),
                    preserve_line_breaks=True,
                    preserve_tabs=True,
                ),
                strategy=ControlCharacterRemovalPreprocessor(),
            ),
        )
    )

    def _stream() -> Iterator[PartitionedRecord]:
        record = ClassificationRecord(
            text="Hello\x00World\nThis\tis\rtest\u200b\x1b!",  # \x00, \x1b, \u200b are controls (wait, \u200b is Cf).
            # Wait, \u200b is Zero Width Space, category Cf (Format). The requirement says 'Cc' (Control).
            # The prompt says "Remove Unicode control characters using Unicode character categories (e.g. unicodedata.category())."
            # It also says: "return unicodedata.category(char) != "Cc" or char in allowed_controls". So only Cc is removed.
            # \x00 and \x1b are Cc. \n, \r, \t are Cc but preserved.
            label="SUPPORTS",
            provenance=ProvenanceMetadata(record_index=1),
        )
        yield PartitionedRecord(partition=PartitionId(name="train"), record=record)

    res = list(DatasetPreprocessor().preprocess(_stream(), pipeline))
    r = typing.cast(ClassificationRecord, res[0].record)
    assert r.text == "HelloWorld\nThis\tis\rtest\u200b!"


def test_control_character_removal_aggressive() -> None:
    """Test removal of line breaks and tabs when not preserved."""
    pipeline = PreprocessingPipeline(
        steps=(
            PreprocessingStep(
                definition=ControlCharacterRemovalDefinition(
                    selectors=(SimpleFieldSelector(field_name="text"),),
                    preserve_line_breaks=False,
                    preserve_tabs=False,
                ),
                strategy=ControlCharacterRemovalPreprocessor(),
            ),
        )
    )

    def _stream() -> Iterator[PartitionedRecord]:
        record = ClassificationRecord(
            text="Hello\nWorld\t!",
            label="SUPPORTS",
            provenance=ProvenanceMetadata(record_index=1),
        )
        yield PartitionedRecord(partition=PartitionId(name="train"), record=record)

    res = list(DatasetPreprocessor().preprocess(_stream(), pipeline))
    r = typing.cast(ClassificationRecord, res[0].record)
    assert r.text == "HelloWorld!"


def test_control_character_removal_validation() -> None:
    with pytest.raises(
        PreprocessingConfigurationError,
        match="ControlCharacterRemovalPreprocessor requires a ControlCharacterRemovalDefinition.",
    ):
        PreprocessingStep(
            definition=PassThroughPreprocessingDefinition(),
            strategy=ControlCharacterRemovalPreprocessor(),
        )
