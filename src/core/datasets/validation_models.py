"""Immutable validation models for constraints and reporting."""

from __future__ import annotations

import re
import typing
from enum import Enum
from typing import Annotated, Mapping, Pattern

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from src.core.datasets.selectors import FieldSelector
from src.core.exceptions import (
    DuplicateValidationProfileError,
    ValidationProfileNotFoundError,
)


class ValidationFailureCode(str, Enum):
    """Strongly typed error codes for validation failures."""

    MISSING_FILE = "MISSING_FILE"
    UNREADABLE = "UNREADABLE"
    INSUFFICIENT_SIZE = "INSUFFICIENT_SIZE"
    MANIFEST_CORRUPT = "MANIFEST_CORRUPT"


class ValidationConstraint(BaseModel):
    """Base class for validation constraints."""

    id: str = Field(..., description="Unique identifier for the constraint")
    target_path: str = Field(..., description="Relative path to the artifact")

    model_config = ConfigDict(frozen=True)


class FileConstraint(ValidationConstraint):
    """Asserts that a file exists, is readable, and optionally meets a size requirement."""

    min_size_bytes: int | None = Field(
        default=None, description="Minimum byte size if required"
    )


class ManifestConstraint(ValidationConstraint):
    """Asserts that a manifest file exists and is valid JSON/JSONL."""

    is_jsonl: bool = Field(
        default=False, description="If true, parse as JSONL instead of JSON"
    )


class ConstraintResult(BaseModel):
    """The evaluated outcome of a single validation constraint."""

    constraint_id: str = Field(..., description="The ID of the constraint evaluated")
    target_path: str = Field(..., description="The target artifact path")
    passed: bool = Field(..., description="Whether the constraint passed")
    failure_code: ValidationFailureCode | None = Field(
        default=None, description="The strongly typed failure code if it failed"
    )

    model_config = ConfigDict(frozen=True)


class ValidationReport(BaseModel):
    """Deterministic, machine-readable validation report."""

    dataset_id: str = Field(..., description="The dataset identity evaluated")
    version: str = Field(..., description="The dataset version evaluated")
    is_valid: bool = Field(..., description="True if all constraints passed")
    results: tuple[ConstraintResult, ...] = Field(
        ..., description="Results for all evaluated constraints"
    )

    model_config = ConfigDict(frozen=True)


class ValidationDefinition(BaseModel):
    """Immutable base model for declarative validation parameters."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class RequiredFieldValidationDefinition(ValidationDefinition):
    """Configuration describing required fields."""

    selectors: tuple[FieldSelector, ...] = Field(
        ..., description="Fields that must resolve to non-null values."
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class EmptyTextValidationDefinition(ValidationDefinition):
    """Configuration describing required non-empty text fields."""

    selectors: tuple[FieldSelector, ...] = Field(
        ..., description="Fields that must resolve to non-empty strings."
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class LengthValidationDefinition(ValidationDefinition):
    """Configuration describing required length constraints for text fields."""

    selectors: tuple[FieldSelector, ...] = Field(
        ..., description="Fields that must satisfy the length constraints."
    )
    min_length: Annotated[int, Field(ge=0)] | None = Field(
        default=None, description="Minimum allowed length."
    )
    max_length: Annotated[int, Field(ge=0)] | None = Field(
        default=None, description="Maximum allowed length."
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _validate_bounds(self) -> "LengthValidationDefinition":
        if self.min_length is None and self.max_length is None:
            # We raise ValueError instead of ValidationConfigurationError here because
            # this is a pydantic model constraint. Pydantic handles ValueError natively.
            raise ValueError(
                "At least one of min_length or max_length must be configured."
            )

        if self.min_length is not None and self.max_length is not None:
            if self.min_length > self.max_length:
                raise ValueError(
                    "min_length cannot be strictly greater than max_length."
                )

        return self


class LabelValidationDefinition(ValidationDefinition):
    """Configuration describing required label validation constraints."""

    selectors: tuple[FieldSelector, ...] = Field(
        ...,
        description="Fields that must belong to the allowed labels set.",
        min_length=1,
    )

    allowed_labels: frozenset[str] = Field(
        ..., description="Immutable collection of valid labels.", min_length=1
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class RegexMatchMode(str, Enum):
    FULLMATCH = "fullmatch"
    MATCH = "match"
    SEARCH = "search"


class RegexValidationDefinition(ValidationDefinition):
    """Configuration describing required regex match constraints for text fields."""

    selectors: tuple[FieldSelector, ...] = Field(
        ..., description="Fields that must satisfy the regex pattern.", min_length=1
    )

    pattern: str = Field(
        ..., description="The regular expression pattern.", min_length=1
    )

    match_mode: RegexMatchMode = Field(
        default=RegexMatchMode.FULLMATCH,
        description="The mode used to match the pattern.",
    )

    _compiled_pattern: Pattern[str] = PrivateAttr()

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _compile_regex(self) -> "RegexValidationDefinition":
        try:
            self._compiled_pattern = re.compile(self.pattern)
        except re.error as e:
            raise ValueError(f"Invalid regular expression pattern: {e}")
        return self


class ValidationProfile(BaseModel):
    """Immutable reusable wrapper representing a pre-configured ValidationPipeline."""

    profile_id: str = Field(
        ..., description="Unique identifier for this validation profile."
    )
    pipeline: ValidationPipeline = Field(
        ...,
        description="The strictly immutable validation pipeline executing this profile.",
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class ValidationProfileRegistry(BaseModel):
    """Immutable namespace for securely resolving named validation profiles."""

    profiles: tuple[ValidationProfile, ...] = Field(
        ..., description="The collection of registered validation profiles."
    )

    _profile_index: Mapping[str, ValidationProfile] = PrivateAttr(default_factory=dict)

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _build_index(self) -> "ValidationProfileRegistry":
        index = {}
        for profile in self.profiles:
            if profile.profile_id in index:
                raise DuplicateValidationProfileError(
                    f"Duplicate validation profile identifier: {profile.profile_id}"
                )
            index[profile.profile_id] = profile

        object.__setattr__(self, "_profile_index", index)
        return self

    def resolve(self, profile_id: str) -> ValidationProfile:
        """Resolves a profile by identifier."""
        if profile_id not in self._profile_index:
            raise ValidationProfileNotFoundError(
                f"Validation profile not found: {profile_id}"
            )
        return self._profile_index[profile_id]


if typing.TYPE_CHECKING:
    from src.core.datasets.validation.base import BaseValidator
else:
    BaseValidator = typing.Any


class ValidationStep(BaseModel):
    """Immutable bound executable defining one exact verification constraint."""

    definition: ValidationDefinition
    strategy: BaseValidator

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _validate_compatibility(self) -> "ValidationStep":
        self.strategy.validate_compatibility(self.definition)
        return self


class ValidationPipeline(BaseModel):
    """Ordered pipeline of configured validation steps."""

    steps: tuple[ValidationStep, ...]

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
