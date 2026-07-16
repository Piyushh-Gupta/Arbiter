"""Unit tests for Arbiter's M4.1 validation framework."""

import types
from collections.abc import Iterator

import pytest
from pydantic import ValidationError

from src.core.datasets.mapping_models import ClassificationRecord
from src.core.datasets.normalization_models import ProvenanceMetadata
from src.core.datasets.partitioning_models import PartitionId
from src.core.datasets.preprocessing_models import (
    PreprocessedRecord,
    PreprocessingMetadata,
)
from src.core.datasets.validation.base import BaseValidator
from src.core.datasets.validation_models import (
    ValidationDefinition,
    ValidationPipeline,
    ValidationStep,
)
from src.core.datasets.validator import DatasetValidator
from src.core.exceptions import (
    DatasetValidationError,
    ValidationConfigurationError,
    ValidationExecutionError,
)


class MockValidationDefinition(ValidationDefinition):
    """Mock declarative parameters."""

    field_name: str
    fail_on: str | None = None
    throw_unexpected: bool = False


class MockValidator(BaseValidator):
    """Mock executable validator."""

    def validate_stream(
        self, stream: Iterator[PreprocessedRecord], definition: ValidationDefinition
    ) -> Iterator[PreprocessedRecord]:
        assert isinstance(definition, MockValidationDefinition)
        for record in stream:
            # We explicitly test stream isolation.
            if definition.throw_unexpected:
                raise RuntimeError("Unexpected failure during execution")

            val = getattr(record.record, definition.field_name, "")
            if definition.fail_on and val == definition.fail_on:
                raise DatasetValidationError(f"Validation failed on {val}")

            yield record

    def validate_compatibility(self, definition: ValidationDefinition) -> None:
        if not isinstance(definition, MockValidationDefinition):
            raise ValidationConfigurationError(
                "Definition must be MockValidationDefinition."
            )


def _create_dummy_record(text: str) -> PreprocessedRecord:
    return PreprocessedRecord(
        record=ClassificationRecord(
            provenance=ProvenanceMetadata(record_index=0),
            text=text,
            label=None,
        ),
        partition=PartitionId(name="train"),
        preprocessing_metadata=PreprocessingMetadata(),
    )


def test_validation_definition_immutability() -> None:
    """Ensure ValidationDefinition is frozen."""
    definition = MockValidationDefinition(field_name="test")
    with pytest.raises(ValidationError):
        definition.field_name = "mutated"


def test_validation_step_immutability() -> None:
    """Ensure ValidationStep is frozen."""
    step = ValidationStep(
        definition=MockValidationDefinition(field_name="test"),
        strategy=MockValidator(),
    )
    with pytest.raises(ValidationError):
        step.definition = MockValidationDefinition(field_name="mutated")


def test_validation_pipeline_immutability() -> None:
    """Ensure ValidationPipeline is frozen."""
    pipeline = ValidationPipeline(steps=())
    with pytest.raises(ValidationError):
        pipeline.steps = ()


def test_compatibility_validation_failure() -> None:
    """Ensure mismatched definition and strategy throws early ValidationConfigurationError."""

    class IncompatibleDefinition(ValidationDefinition):
        pass

    with pytest.raises(ValidationConfigurationError):
        ValidationStep(
            definition=IncompatibleDefinition(),
            strategy=MockValidator(),
        )


def test_successful_passthrough_and_object_identity() -> None:
    """Ensure the validator perfectly yields exact identical objects continuously."""
    records = [
        _create_dummy_record("hello"),
        _create_dummy_record("world"),
    ]

    pipeline = ValidationPipeline(
        steps=(
            ValidationStep(
                definition=MockValidationDefinition(field_name="text"),
                strategy=MockValidator(),
            ),
        )
    )

    validator = DatasetValidator()
    stream = validator.validate(iter(records), pipeline)

    # Iterator checks
    assert isinstance(stream, types.GeneratorType)

    results = list(stream)
    assert len(results) == 2

    # Exact Object Identity Preserved!
    assert id(results[0]) == id(records[0])
    assert id(results[1]) == id(records[1])


def test_fail_fast_validation_execution() -> None:
    """Ensure the execution immediately stops mapping when hitting DatasetValidationError."""
    records = [
        _create_dummy_record("valid_1"),
        _create_dummy_record("invalid_val"),
        _create_dummy_record("valid_2"),
    ]

    pipeline = ValidationPipeline(
        steps=(
            ValidationStep(
                definition=MockValidationDefinition(
                    field_name="text", fail_on="invalid_val"
                ),
                strategy=MockValidator(),
            ),
        )
    )

    validator = DatasetValidator()
    stream = validator.validate(iter(records), pipeline)

    # Valid string correctly mapped natively exactly identically seamlessly.
    r1 = next(stream)
    assert id(r1) == id(records[0])

    # Next iteration must dynamically short-circuit fail-fast exclusively mapping.
    with pytest.raises(
        DatasetValidationError, match="Validation failed on invalid_val"
    ):
        next(stream)


def test_unexpected_error_wrapped() -> None:
    """Ensure any non-domain exception is structurally wrapped correctly cleanly."""
    records = [
        _create_dummy_record("valid"),
    ]

    pipeline = ValidationPipeline(
        steps=(
            ValidationStep(
                definition=MockValidationDefinition(
                    field_name="text", throw_unexpected=True
                ),
                strategy=MockValidator(),
            ),
        )
    )

    validator = DatasetValidator()
    stream = validator.validate(iter(records), pipeline)

    with pytest.raises(
        ValidationExecutionError,
        match="Unexpected failure during dataset validation execution: Unexpected failure",
    ):
        list(stream)
