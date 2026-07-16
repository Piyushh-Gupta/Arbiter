"""Data models and predicates for the filtering and selection layer."""

import typing
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from src.core.datasets.filtering.base import BaseFilter
from src.core.datasets.selectors import FieldSelector


class FieldExistsPredicate(BaseModel):
    """Standalone declarative configuration ensuring a field is not None."""

    selector: FieldSelector

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class FieldEqualsPredicate(BaseModel):
    """Standalone declarative configuration for exact value equivalence."""

    selector: FieldSelector
    target_value: str | int | float | bool

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class FieldInSetPredicate(BaseModel):
    """Standalone declarative configuration for subset inclusion."""

    selector: FieldSelector
    allowed_values: frozenset[str | int | float | bool]

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class FieldLengthPredicate(BaseModel):
    """Standalone declarative configuration for length boundaries."""

    selector: FieldSelector
    min_length: int | None = None
    max_length: int | None = None

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


@dataclass(frozen=True)
class FilterStep:
    """Unified, immutable representation of a resolved filter configuration and execution strategy."""

    definition: typing.Any
    filter_strategy: BaseFilter


class FilterPipeline(BaseModel):
    """Immutable ordered execution chain."""

    steps: tuple[FilterStep, ...]

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
