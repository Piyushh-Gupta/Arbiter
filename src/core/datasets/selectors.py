"""Canonical shared field resolution abstractions."""

import typing

from pydantic import BaseModel, ConfigDict

from src.core.datasets.mapping_models import TaskRecord
from src.core.exceptions import FieldResolutionError


@typing.runtime_checkable
class FieldSelector(typing.Protocol):
    """
    Abstract decoupled layer governing exactly how attributes are extracted
    and replaced from canonical task records.
    """

    def resolve(self, record: TaskRecord) -> typing.Any: ...
    def replace(self, record: TaskRecord, new_value: typing.Any) -> TaskRecord: ...


class SimpleFieldSelector(BaseModel):
    """Safely extracts and replaces top-level structural fields from a TaskRecord."""

    field_name: str

    model_config = ConfigDict(frozen=True)

    def resolve(self, record: TaskRecord) -> typing.Any:
        if not hasattr(record, self.field_name):
            raise FieldResolutionError(
                f"Attribute '{self.field_name}' not found on record."
            )
        return getattr(record, self.field_name)

    def replace(self, record: TaskRecord, new_value: typing.Any) -> TaskRecord:
        if not hasattr(record, self.field_name):
            raise FieldResolutionError(
                f"Attribute '{self.field_name}' not found on record."
            )
        return record.model_copy(update={self.field_name: new_value})
