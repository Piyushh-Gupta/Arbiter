"""Base protocols for the serialization framework."""

from typing import Iterator, Protocol

from src.core.datasets.preprocessing_models import PreprocessedRecord
from src.core.datasets.serialization_models import SerializationDefinition


class BaseSerializer(Protocol):
    """Protocol defining the structural contract for all dataset serializers."""

    def serialize_stream(
        self,
        stream: Iterator[PreprocessedRecord],
        definition: SerializationDefinition,
    ) -> Iterator[PreprocessedRecord]:
        """
        Consumes the stream to serialize records while yielding identical records for downstream steps.

        Must strictly preserve object identity and never buffer records.
        """
        ...

    def validate_compatibility(
        self,
        definition: SerializationDefinition,
    ) -> None:
        """
        Statically enforces compatibility between this strategy and the provided definition.

        Raises SerializationConfigurationError if incompatible.
        """
        ...
