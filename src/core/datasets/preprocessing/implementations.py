"""Stateless preprocessing implementations."""

import typing
from collections.abc import Iterator

from src.core.datasets.preprocessing_models import (
    PassThroughPreprocessingDefinition,
    PreprocessedRecord,
    PreprocessingDefinition,
)
from src.core.exceptions import PreprocessingConfigurationError


class PassThroughPreprocessor:
    """Baseline deterministic implementation that yields records completely unaltered."""

    def process_stream(
        self, stream: Iterator[PreprocessedRecord], definition: PreprocessingDefinition
    ) -> Iterator[PreprocessedRecord]:
        _ = typing.cast(PassThroughPreprocessingDefinition, definition)
        for record in stream:
            # Explicitly demonstrating the transformation contract (though here it's purely a pass-through)
            # Future processors will yield PreprocessedRecord(
            #     partition=record.partition,
            #     record=record.record,
            #     preprocessing_metadata=new_metadata
            # )
            yield record

    def validate_compatibility(self, definition: PreprocessingDefinition) -> None:
        """Ensure definition matches execution prerequisites."""
        if not isinstance(definition, PassThroughPreprocessingDefinition):
            raise PreprocessingConfigurationError(
                "PassThroughPreprocessor requires a PassThroughPreprocessingDefinition."
            )
