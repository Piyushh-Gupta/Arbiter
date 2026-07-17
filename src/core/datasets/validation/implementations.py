"""Concrete implementations for dataset validation constraints."""

from collections.abc import Iterator

from src.core.datasets.preprocessing_models import PreprocessedRecord
from src.core.datasets.validation.base import BaseValidator
from src.core.datasets.validation_models import (
    EmptyTextValidationDefinition,
    LengthValidationDefinition,
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


class EmptyTextValidator(BaseValidator):
    """Stateless validator verifying that text fields are not empty."""

    def validate_stream(
        self,
        stream: Iterator[PreprocessedRecord],
        definition: ValidationDefinition,
    ) -> Iterator[PreprocessedRecord]:
        """
        Validates records ensuring configured text fields contain non-empty text.

        Propagates FieldResolutionError unchanged for structural schema mismatch.
        Assumes the resolved value is textual and has a .strip() method.
        Raises DatasetValidationError strictly on encountering empty text after stripping.
        Preserves object identity identically through the pipeline natively.
        """
        assert isinstance(definition, EmptyTextValidationDefinition)

        for preprocessed_record in stream:
            for selector in definition.selectors:
                # Let FieldResolutionError natively propagate upward cleanly
                value = selector.resolve(preprocessed_record.record)

                if value.strip() == "":
                    raise DatasetValidationError(
                        f"Validation failed: Mandatory text field evaluated to empty text via {selector}"
                    )

            # Preserve identical underlying immutable object exactly passing it downward
            yield preprocessed_record

    def validate_compatibility(self, definition: ValidationDefinition) -> None:
        """Enforces strictly bound execution parameters at model construction statically."""
        if not isinstance(definition, EmptyTextValidationDefinition):
            raise ValidationConfigurationError(
                "EmptyTextValidator requires an EmptyTextValidationDefinition."
            )


class LengthValidator(BaseValidator):
    """Stateless validator verifying that text fields satisfy configured length bounds."""

    def validate_stream(
        self,
        stream: Iterator[PreprocessedRecord],
        definition: ValidationDefinition,
    ) -> Iterator[PreprocessedRecord]:
        """
        Validates records ensuring configured text fields meet length bounds.

        Propagates FieldResolutionError unchanged for structural schema mismatch.
        Raises DatasetValidationError strictly on encountering length violations.
        Preserves object identity identically through the pipeline natively.
        """
        assert isinstance(definition, LengthValidationDefinition)

        for preprocessed_record in stream:
            for selector in definition.selectors:
                value = selector.resolve(preprocessed_record.record)

                val_len = len(value)

                if (
                    definition.min_length is not None
                    and val_len < definition.min_length
                ):
                    raise DatasetValidationError(
                        f"Validation failed: Length {val_len} is strictly below minimum {definition.min_length} via {selector}"
                    )

                if (
                    definition.max_length is not None
                    and val_len > definition.max_length
                ):
                    raise DatasetValidationError(
                        f"Validation failed: Length {val_len} is strictly above maximum {definition.max_length} via {selector}"
                    )

            # Preserve identical underlying immutable object exactly passing it downward
            yield preprocessed_record

    def validate_compatibility(self, definition: ValidationDefinition) -> None:
        """Enforces strictly bound execution parameters at model construction statically."""
        if not isinstance(definition, LengthValidationDefinition):
            raise ValidationConfigurationError(
                "LengthValidator requires a LengthValidationDefinition."
            )
