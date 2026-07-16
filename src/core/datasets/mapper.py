"""Dataset orchestrator for schema mapping."""

import logging
from collections.abc import Iterator

from src.core.datasets.mapping.registry import MappingRegistry
from src.core.datasets.mapping_models import TaskRecord, TaskSchemaType
from src.core.datasets.normalization_models import NormalizedRecord
from src.core.exceptions import MissingRequiredFieldError, SchemaMappingError

logger = logging.getLogger(__name__)


class DatasetMapper:
    """
    Stateless service orchestrating the schema mapping layer.

    Consumes canonical NormalizedRecords and delegating the transformation
    to the strategy bound within the MappingRegistry.
    """

    def __init__(self, registry: MappingRegistry) -> None:
        self._registry = registry

    def map(
        self, stream: Iterator[NormalizedRecord], task_type: TaskSchemaType
    ) -> Iterator[TaskRecord]:
        """
        Maps a stream of generic canonical records into task-specific records.

        Args:
            stream: A lazy generator yielding NormalizedRecord instances.
            task_type: The task schema type to map into.

        Yields:
            Immutable TaskRecord instances mapped downstream.

        Raises:
            SchemaMappingError: If schema mapping fails unexpectedly.
        """
        logger.info(f"Starting schema mapping for task: {task_type.value}")
        task_mapping = self._registry.resolve(task_type)

        try:
            count = 0
            # Delegates execution by calling the mapper with the associated schema definition
            for count, record in enumerate(
                task_mapping.mapper.map_stream(stream, task_mapping.schema_definition),
                start=1,
            ):
                yield record

            logger.info(
                f"Schema mapping completed successfully. Processed {count} records."
            )
        except MissingRequiredFieldError:
            raise
        except Exception as e:
            logger.error("Schema mapping failure encountered.", exc_info=True)
            raise SchemaMappingError("Schema mapping failed") from e
