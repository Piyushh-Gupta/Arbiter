"""Concrete implementations for dataset validation constraints."""

from collections.abc import Iterator

from src.core.datasets.preprocessing_models import PreprocessedRecord
from src.core.datasets.validation.base import BaseValidator
from src.core.datasets.validation_models import (
    RequiredFieldValidationDefinition,
    ValidationDefinition,
)
from src.core.exceptions import DatasetValidationError, ValidationConfigurationError


class RequiredFieldValidator(BaseValidator):
    """Stateless validator verifying that required fields exist and are populated."""

    def validate_stream(
        self,
        stream: Iterator[PreprocessedRecord],
        definition: ValidationDefinition,
    ) -> Iterator[PreprocessedRecord]:
        """
        Validates records against the configured required fields.

        Propagates FieldResolutionError unchanged for structural schema mismatch.
        Raises DatasetValidationError strictly on encountering explicit None values.
        Preserves object identity identically through the pipeline natively.
        """
        assert isinstance(definition, RequiredFieldValidationDefinition)

        for preprocessed_record in stream:
            for selector in definition.selectors:
                # Let FieldResolutionError natively propagate upward cleanly
                value = selector.resolve(preprocessed_record.record)

                if value is None:
                    raise DatasetValidationError(
                        f"Validation failed: Mandatory field evaluated to None via {selector}"
                    )

            # Preserve identical underlying immutable object exactly passing it downward
            yield preprocessed_record

    def validate_compatibility(self, definition: ValidationDefinition) -> None:
        """Enforces strictly bound execution parameters at model construction statically."""
        if not isinstance(definition, RequiredFieldValidationDefinition):
            raise ValidationConfigurationError(
                "RequiredFieldValidator requires a RequiredFieldValidationDefinition."
            )
