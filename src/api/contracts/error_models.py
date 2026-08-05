"""Standardized error models for the API Contract layer."""

from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field


class ValidationErrorDetail(BaseModel):
    """Immutable model representing a single field-level validation error."""

    loc: tuple[str | int, ...] = Field(
        ..., description="Location of the error in the payload"
    )
    msg: str = Field(..., description="Human-readable error message")
    type: str = Field(..., description="Error type identifier")

    model_config = ConfigDict(frozen=True, extra="forbid")


class ErrorEnvelope(BaseModel):
    """Immutable envelope for standardizing API error responses."""

    error_code: str = Field(..., description="A unique machine-readable error code")
    message: str = Field(..., description="A human-readable error description")
    correlation_id: str | None = Field(
        default=None, description="Request correlation ID"
    )
    details: list[ValidationErrorDetail] | Mapping[str, Any] | None = Field(
        default=None, description="Optional detailed error context"
    )

    model_config = ConfigDict(frozen=True, extra="forbid")
