"""Dataset orchestration layer for dataset filtering."""

import logging
from collections.abc import Iterator

from src.core.datasets.filtering_models import FilterPipeline
from src.core.datasets.mapping_models import TaskRecord
from src.core.exceptions import (
    FieldResolutionError,
    FilterConfigurationError,
    FilterExecutionError,
)

logger = logging.getLogger(__name__)


class DatasetFilter:
    """
    Stateless orchestration service applying pipelines lazily.
    """

    def filter(
        self, stream: Iterator[TaskRecord], pipeline: FilterPipeline
    ) -> Iterator[TaskRecord]:
        """
        Executes a pipeline of filters sequentially over a task record stream.

        Args:
            stream: Incoming stream of TaskRecords.
            pipeline: Ordered pipeline of immutable filter steps.

        Yields:
            Immutable TaskRecords that have passed all filters.

        Raises:
            FieldResolutionError: If a FieldSelector cannot extract an attribute.
            FilterConfigurationError: If a step is invalidly configured.
            FilterExecutionError: If an unexpected execution failure occurs.
        """
        logger.info(
            f"Starting filter pipeline execution with {len(pipeline.steps)} sequential filters."
        )

        try:
            current_stream = stream
            for step in pipeline.steps:
                current_stream = step.filter_strategy.filter_stream(
                    current_stream, step.definition
                )

            accepted_count = 0
            for record in current_stream:
                accepted_count += 1
                yield record

            logger.info(
                f"Filter pipeline completed. {accepted_count} records accepted downstream."
            )
        except (FieldResolutionError, FilterConfigurationError):
            # Domain exceptions propagate unchanged natively
            raise
        except Exception as e:
            logger.error("Filter pipeline execution failed.", exc_info=True)
            raise FilterExecutionError("Filter execution failed") from e
