"""Abstract base protocols and mapping for the preprocessing layer."""

import typing
from collections.abc import Iterator
from dataclasses import dataclass

from src.core.datasets.preprocessing_models import (
    PreprocessedRecord,
    PreprocessingDefinition,
)


@typing.runtime_checkable
class BasePreprocessor(typing.Protocol):
    """Stateless protocol dictating sequence transformation logic."""

    def process_stream(
        self, stream: Iterator[PreprocessedRecord], definition: PreprocessingDefinition
    ) -> Iterator[PreprocessedRecord]: ...

    def validate_compatibility(self, definition: PreprocessingDefinition) -> None:
        """Strongly typed validation ensuring the definition matches execution prerequisites."""
        ...


@dataclass(frozen=True)
class PreprocessingStep:
    """Immutable bound executable defining one exact transformation pass."""

    definition: PreprocessingDefinition
    strategy: BasePreprocessor

    def __post_init__(self) -> None:
        """Enforces strongly typed configuration compatibility at instantiation."""
        self.strategy.validate_compatibility(self.definition)


@dataclass(frozen=True)
class PreprocessingPipeline:
    """Ordered pipeline of configured preprocessing steps."""

    steps: tuple[PreprocessingStep, ...]
