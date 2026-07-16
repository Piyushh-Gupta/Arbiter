"""Stateless assignment implementations."""

import hashlib
import typing

from src.core.datasets.mapping_models import TaskRecord
from src.core.datasets.partitioning_models import (
    HashPartitionDefinition,
    ModuloPartitionDefinition,
    PartitionDefinition,
    PartitionId,
)
from src.core.exceptions import PartitionAssignmentError


class ModuloPartitioner:
    """Stateless assignment using integer modulo. Runs completely unprotected relying on safe bootstrap validations."""

    def assign(
        self, record: TaskRecord, definition: PartitionDefinition
    ) -> PartitionId | None:
        # Cast explicitly based on the formal guarantee enforced during mapping bootstrap
        modulo_def = typing.cast(ModuloPartitionDefinition, definition)

        value = modulo_def.selector.resolve(record)
        if not isinstance(value, int):
            raise PartitionAssignmentError(
                "Modulo partitioner requires an integer field."
            )

        remainder = value % modulo_def.modulo_divisor
        for partition_id, remainders in modulo_def.partition_map.items():
            if remainder in remainders:
                return partition_id
        return None


class HashPartitioner:
    """Stateless assignment using stable hashing of string fields."""

    def assign(
        self, record: TaskRecord, definition: PartitionDefinition
    ) -> PartitionId | None:
        hash_def = typing.cast(HashPartitionDefinition, definition)

        value = hash_def.selector.resolve(record)
        if not isinstance(value, str):
            raise PartitionAssignmentError("Hash partitioner requires a string field.")

        # Stable deterministic hash using MD5
        hash_int = int(hashlib.md5(value.encode("utf-8")).hexdigest(), 16)
        bucket = hash_int % hash_def.hash_buckets

        for partition_id, buckets in hash_def.partition_map.items():
            if bucket in buckets:
                return partition_id
        return None
