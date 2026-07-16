"""Dataset orchestration layer for preprocessing."""

import logging
from collections.abc import Iterator

from src.core.datasets.partitioning_models import PartitionedRecord
from src.core.datasets.preprocessing.base import PreprocessingPipeline
from src.core.datasets.preprocessing_models import (
    PreprocessedRecord,
    PreprocessingMetadata,
)
from src.core.exceptions import (
    PreprocessingConfigurationError,
    PreprocessingExecutionError,
)

logger = logging.getLogger(__name__)


class DatasetPreprocessor:
    """Stateless orchestrator that evaluates an ordered preprocessing pipeline over streaming data."""

    def preprocess(
        self, stream: Iterator[PartitionedRecord], pipeline: PreprocessingPipeline
    ) -> Iterator[PreprocessedRecord]:
        """
        Transforms an incoming stream of partitioned records into a preprocessed stream.

        Args:
            stream: A lazy stream of immutable PartitionedRecords.
            pipeline: An ordered pipeline of configured preprocessing steps.

        Yields:
            Immutable PreprocessedRecord instances containing the target PartitionId, TaskRecord, and Metadata.

        Raises:
            PreprocessingConfigurationError: If configuration mismatch happens during execution.
            PreprocessingExecutionError: If an unexpected execution failure occurs.
        """
        logger.info(
            f"Starting dataset preprocessing pipeline containing {len(pipeline.steps)} steps."
        )

        try:
            # Baseline Conversion: Safely map PartitionedRecord to empty PreprocessedRecord
            # Maintains identical memory pointers for the underlying TaskRecord
            current_stream: Iterator[PreprocessedRecord] = (
                PreprocessedRecord(
                    partition=r.partition,
                    record=r.record,
                    preprocessing_metadata=PreprocessingMetadata(),
                )
                for r in stream
            )

            # Lazy Generator Chaining
            for step in pipeline.steps:
                current_stream = step.strategy.process_stream(
                    current_stream, step.definition
                )

            processed_count = 0
            for record in current_stream:
                processed_count += 1
                yield record

            logger.info(
                f"Preprocessing pipeline completed successfully. Evaluated {processed_count} records."
            )

        except PreprocessingConfigurationError:
            raise
        except Exception as e:
            logger.error(
                "Failed executing dataset preprocessing pipeline.", exc_info=True
            )
            raise PreprocessingExecutionError("Pipeline execution failed") from e
