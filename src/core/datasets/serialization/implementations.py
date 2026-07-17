"""Concrete implementations of the serialization framework strategies."""

import json
from typing import Iterator, cast

from src.core.datasets.preprocessing_models import PreprocessedRecord
from src.core.datasets.serialization.base import BaseSerializer
from src.core.datasets.serialization_models import (
    JsonlSerializationDefinition,
    ManifestSerializationDefinition,
    MetadataSerializationDefinition,
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


class MetadataSerializer(BaseSerializer):
    """Deterministically serializes externally supplied dataset metadata without buffering records."""

    def serialize_stream(
        self,
        stream: Iterator[PreprocessedRecord],
        definition: SerializationDefinition,
    ) -> Iterator[PreprocessedRecord]:
        # Compatibility is guaranteed by the framework; perform typed cast
        config = cast(MetadataSerializationDefinition, definition)

        # Write metadata exactly once before the first record is consumed
        # Intentional Formatting Policy: Pretty-printed human-readable JSON
        with config.output_path.open(
            mode="w", encoding=config.encoding, newline="\n"
        ) as f:
            json_string = json.dumps(
                config.metadata,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            f.write(f"{json_string}\n")

        # Stream propagation cleanly efficiently natively
        for record in stream:
            # Strict object identity preservation
            yield record

    def validate_compatibility(self, definition: SerializationDefinition) -> None:
        if not isinstance(definition, MetadataSerializationDefinition):
            raise SerializationConfigurationError(
                f"MetadataSerializer requires a MetadataSerializationDefinition, got {type(definition).__name__}"
            )


class ManifestSerializer(BaseSerializer):
    """Deterministically persists an immutable dataset manifest without buffering records."""

    def serialize_stream(
        self,
        stream: Iterator[PreprocessedRecord],
        definition: SerializationDefinition,
    ) -> Iterator[PreprocessedRecord]:
        # Compatibility is guaranteed by the framework; perform typed cast
        config = cast(ManifestSerializationDefinition, definition)

        # Write manifest exactly once before the first record is consumed
        # Formatting Policy: Pretty-printed human-readable JSON, consistent with MetadataSerializer
        with config.output_path.open(
            mode="w", encoding=config.encoding, newline="\n"
        ) as f:
            raw_dict = config.manifest.model_dump(mode="json")
            json_string = json.dumps(
                raw_dict, ensure_ascii=False, indent=2, sort_keys=True
            )
            f.write(f"{json_string}\n")

        for record in stream:
            # Strict object identity preservation
            yield record

    def validate_compatibility(self, definition: SerializationDefinition) -> None:
        if not isinstance(definition, ManifestSerializationDefinition):
            raise SerializationConfigurationError(
                f"ManifestSerializer requires a ManifestSerializationDefinition, "
                f"got {type(definition).__name__}"
            )
