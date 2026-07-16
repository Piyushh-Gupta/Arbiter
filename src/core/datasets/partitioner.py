"""Dataset orchestration layer for partitioning."""

import logging
from collections.abc import Iterator

from src.core.datasets.mapping_models import TaskRecord
from src.core.datasets.partitioning.base import PartitionMapping
from src.core.datasets.partitioning_models import PartitionedRecord
from src.core.exceptions import PartitionAssignmentError, PartitionExecutionError

logger = logging.getLogger(__name__)


class DatasetPartitioner:
    """Stateless orchestration service applying unified partition mappings lazily."""

    def partition(
        self, stream: Iterator[TaskRecord], mapping: PartitionMapping
    ) -> Iterator[PartitionedRecord]:
        """
        Transforms an incoming stream of task records into a partitioned stream.

        Records that are assigned a partition are yielded wrapped in a PartitionedRecord.
        Records that do not map to any partition are silently dropped (act as sub-filtered).

        Args:
            stream: A lazy stream of immutable TaskRecords.
            mapping: A unified binding of partition definition and strategy.

        Yields:
            Immutable PartitionedRecord instances containing the target PartitionId and TaskRecord.

        Raises:
            PartitionAssignmentError: If evaluation encounters a mathematical failure (e.g. wrong type).
            PartitionExecutionError: If an unexpected execution failure occurs.
        """
        logger.info("Starting dataset stream partition assignment.")

        try:
            assigned_count = 0
            dropped_count = 0

            for record in stream:
                assigned_partition = mapping.strategy.assign(record, mapping.definition)

                if assigned_partition is None:
                    dropped_count += 1
                    continue

                assigned_count += 1
                yield PartitionedRecord(partition=assigned_partition, record=record)

            logger.info(
                f"Partitioning completed. {assigned_count} records assigned. {dropped_count} records dropped."
            )

        except PartitionAssignmentError:
            # Propagate native domain mapping exceptions explicitly
            raise
        except Exception as e:
            logger.error("Failed executing partitioned dataset stream.", exc_info=True)
            raise PartitionExecutionError("Partition execution failed") from e
