"""Abstract base protocols and mapping for the validation layer."""

import typing
from collections.abc import Iterator

from src.core.datasets.preprocessing_models import PreprocessedRecord
from src.core.datasets.validation_models import ValidationDefinition


@typing.runtime_checkable
class BaseValidator(typing.Protocol):
    """Stateless protocol dictating verification logic."""

    def validate_stream(
        self, stream: Iterator[PreprocessedRecord], definition: ValidationDefinition
    ) -> Iterator[PreprocessedRecord]: ...

    def validate_compatibility(self, definition: ValidationDefinition) -> None:
        """Strongly typed validation ensuring the definition matches execution prerequisites."""
        ...
