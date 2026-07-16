"""Stateless text preprocessing implementations."""

import re
import typing
import unicodedata
from collections.abc import Callable, Iterator

from src.core.datasets.mapping_models import TaskRecord
from src.core.datasets.preprocessing.text_models import (
    UnicodeNormalizationDefinition,
    WhitespaceNormalizationDefinition,
)
from src.core.datasets.preprocessing_models import (
    PreprocessedRecord,
    PreprocessingDefinition,
)
from src.core.datasets.selectors import FieldSelector
from src.core.exceptions import PreprocessingConfigurationError


def _apply_text_transformation(
    record: TaskRecord,
    selectors: tuple[FieldSelector, ...],
    transform_fn: Callable[[str], str],
) -> TaskRecord:
    """
    Shared private implementation utility to safely extract, transform, and replace textual properties.
    Not a standalone framework/orchestrator, purely a code reuse mechanism.
    """
    current_record = record
    for selector in selectors:
        try:
            val = selector.resolve(current_record)
            if isinstance(val, str):
                new_val = transform_fn(val)
                current_record = selector.replace(current_record, new_val)
        except Exception:
            # Explicitly ignore non-target structural missing fields silently
            pass
    return current_record


class WhitespaceNormalizationPreprocessor:
    """Stateless preprocessor ensuring deterministic text spacing natively mapped via shared field selectors."""

    def process_stream(
        self, stream: Iterator[PreprocessedRecord], definition: PreprocessingDefinition
    ) -> Iterator[PreprocessedRecord]:
        config = typing.cast(WhitespaceNormalizationDefinition, definition)

        def _transform(val: str) -> str:
            new_val = val
            if config.trim_leading:
                new_val = new_val.lstrip()
            if config.trim_trailing:
                new_val = new_val.rstrip()
            if config.collapse_multiple:
                new_val = re.sub(r"\s+", " ", new_val)
            return new_val

        for preprocessed_record in stream:
            new_record = _apply_text_transformation(
                record=preprocessed_record.record,
                selectors=config.selectors,
                transform_fn=_transform,
            )

            yield PreprocessedRecord(
                partition=preprocessed_record.partition,
                record=new_record,
                preprocessing_metadata=preprocessed_record.preprocessing_metadata,
            )

    def validate_compatibility(self, definition: PreprocessingDefinition) -> None:
        if not isinstance(definition, WhitespaceNormalizationDefinition):
            raise PreprocessingConfigurationError(
                "WhitespaceNormalizationPreprocessor requires a WhitespaceNormalizationDefinition."
            )


class UnicodeNormalizationPreprocessor:
    """Stateless preprocessor enforcing strict Unicode encoding forms mapped via shared field selectors."""

    def process_stream(
        self, stream: Iterator[PreprocessedRecord], definition: PreprocessingDefinition
    ) -> Iterator[PreprocessedRecord]:
        config = typing.cast(UnicodeNormalizationDefinition, definition)

        def _transform(val: str) -> str:
            return unicodedata.normalize(config.normalization_form, val)

        for preprocessed_record in stream:
            new_record = _apply_text_transformation(
                record=preprocessed_record.record,
                selectors=config.selectors,
                transform_fn=_transform,
            )

            yield PreprocessedRecord(
                partition=preprocessed_record.partition,
                record=new_record,
                preprocessing_metadata=preprocessed_record.preprocessing_metadata,
            )

    def validate_compatibility(self, definition: PreprocessingDefinition) -> None:
        if not isinstance(definition, UnicodeNormalizationDefinition):
            raise PreprocessingConfigurationError(
                "UnicodeNormalizationPreprocessor requires a UnicodeNormalizationDefinition."
            )
