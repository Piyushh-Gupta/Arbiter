"""Immutable configurations for text transformations."""

import typing

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


class UnicodeNormalizationDefinition(BaseModel):
    """Configuration for standardized Unicode normalization."""

    selectors: tuple[FieldSelector, ...] = Field(
        ..., description="Abstract field selectors pointing to target content."
    )
    normalization_form: typing.Literal["NFC", "NFD", "NFKC", "NFKD"] = Field(
        default="NFC", description="The exact Unicode normalization standard to apply."
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
