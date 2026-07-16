"""Immutable validation models for constraints and reporting."""

from __future__ import annotations

import typing
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
