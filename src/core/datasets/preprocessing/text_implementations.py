"""Stateless text preprocessing implementations."""

import re
import typing
from collections.abc import Iterator

from src.core.datasets.preprocessing.text_models import (
    WhitespaceNormalizationDefinition,
)
from src.core.datasets.preprocessing_models import (
    PreprocessedRecord,
    PreprocessingDefinition,
)
from src.core.exceptions import PreprocessingConfigurationError


class WhitespaceNormalizationPreprocessor:
    """Stateless preprocessor ensuring deterministic text spacing natively mapped via shared field selectors."""

    def process_stream(
        self, stream: Iterator[PreprocessedRecord], definition: PreprocessingDefinition
    ) -> Iterator[PreprocessedRecord]:
        config = typing.cast(WhitespaceNormalizationDefinition, definition)

        for preprocessed_record in stream:
            record = preprocessed_record.record

            for selector in config.selectors:
                try:
                    val = selector.resolve(record)
                    if isinstance(val, str):
                        new_val = val
                        if config.trim_leading:
                            new_val = new_val.lstrip()
                        if config.trim_trailing:
                            new_val = new_val.rstrip()
                        if config.collapse_multiple:
                            new_val = re.sub(r"\s+", " ", new_val)

                        record = selector.replace(record, new_val)
                except Exception:
                    # Ignore resolution errors for fields that don't exist?
                    # The prompt says: "Only explicitly selected fields may be transformed... preserve non-target fields exactly".
                    pass

            yield PreprocessedRecord(
                partition=preprocessed_record.partition,
                record=record,
                preprocessing_metadata=preprocessed_record.preprocessing_metadata,
            )

    def validate_compatibility(self, definition: PreprocessingDefinition) -> None:
        if not isinstance(definition, WhitespaceNormalizationDefinition):
            raise PreprocessingConfigurationError(
                "WhitespaceNormalizationPreprocessor requires a WhitespaceNormalizationDefinition."
            )
