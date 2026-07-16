"""Dataset normalization service orchestrating the canonical internal contract."""

import logging
from collections.abc import Iterator

from pydantic import ValidationError

from src.core.datasets.normalization_models import NormalizedRecord, ProvenanceMetadata
from src.core.datasets.parser_models import ParsedRecord
from src.core.exceptions import (
    MalformedNormalizedRecordError,
    NormalizationFailureError,
)

logger = logging.getLogger(__name__)


class DatasetNormalizer:
    """
    Stateless normalization service exposing a singular canonical transformation pipeline.
    """

    def normalize(self, stream: Iterator[ParsedRecord]) -> Iterator[NormalizedRecord]:
        """
        Iterates over syntactic records and converts them to the canonical domain.

        Args:
            stream: A lazy generator yielding ParsedRecord instances.

        Yields:
            Immutable NormalizedRecord instances 1-to-1 enriched with provenance metadata.

        Raises:
            NormalizationFailureError: If transformation fails unexpectedly.
            MalformedNormalizedRecordError: If canonical validation fails.
        """
        logger.info("Starting normalization pipeline.")
        count = 0

        try:
            for count, record in enumerate(stream, start=1):
                provenance = ProvenanceMetadata(record_index=count)
                try:
                    yield NormalizedRecord(content=record.data, provenance=provenance)
                except ValidationError as e:
                    raise MalformedNormalizedRecordError(
                        f"Failed to structural validation for record {count}: {e}"
                    ) from e

            logger.info(
                f"Normalization completed successfully. Processed {count} records."
            )
        except MalformedNormalizedRecordError:
            # Reraise explicitly raised validation errors natively
            raise
        except Exception as e:
            logger.error("Normalization failure encountered.", exc_info=True)
            raise NormalizationFailureError("Normalization pipeline failed") from e
