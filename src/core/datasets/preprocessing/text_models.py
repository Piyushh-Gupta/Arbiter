"""Immutable configurations for text transformations."""

from pydantic import BaseModel, ConfigDict, Field

from src.core.datasets.selectors import FieldSelector


class WhitespaceNormalizationDefinition(BaseModel):
    """Configuration for generic whitespace normalization."""

    selectors: tuple[FieldSelector, ...] = Field(
        ..., description="Abstract field selectors pointing to target content."
    )
    collapse_multiple: bool = True
    trim_leading: bool = True
    trim_trailing: bool = True

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
