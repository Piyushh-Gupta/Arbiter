"""Concrete implementations of the serialization framework strategies."""

import json
from typing import Iterator, cast

from src.core.datasets.preprocessing_models import PreprocessedRecord
from src.core.datasets.serialization.base import BaseSerializer
from src.core.datasets.serialization_models import (
    JsonlSerializationDefinition,
    SerializationDefinition,
)
from src.core.exceptions import SerializationConfigurationError


class JsonlSerializer(BaseSerializer):
    """Deterministically serializes records to a JSON Lines file without buffering."""

    def serialize_stream(
        self,
        stream: Iterator[PreprocessedRecord],
        definition: SerializationDefinition,
    ) -> Iterator[PreprocessedRecord]:
        # Compatibility is guaranteed by the framework; perform typed cast
        config = cast(JsonlSerializationDefinition, definition)

        # Open using Path API with deterministic newlines
        with config.output_path.open(
            mode="w", encoding=config.encoding, newline="\n"
        ) as f:
            for record in stream:
                # Decouple formatting: Extract JSON-compatible dictionary, then format as JSON string
                raw_dict = record.record.model_dump(mode="json", by_alias=True)
                json_string = json.dumps(raw_dict, ensure_ascii=False)

                f.write(f"{json_string}\n")

                # Strict object identity preservation
                yield record

    def validate_compatibility(self, definition: SerializationDefinition) -> None:
        if not isinstance(definition, JsonlSerializationDefinition):
            raise SerializationConfigurationError(
                f"JsonlSerializer requires a JsonlSerializationDefinition, got {type(definition).__name__}"
            )
