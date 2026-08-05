"""Response models for the API Contract layer."""

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class CorrelationMetadata(BaseModel):
    """Immutable model representing correlation context for a request."""

    correlation_id: str = Field(...)
    client_id: str | None = Field(default=None)

    model_config = ConfigDict(frozen=True, extra="forbid")


class ApiResponseEnvelope(BaseModel, Generic[T]):
    """Generic immutable envelope for successful responses."""

    data: T = Field(...)
    metadata: dict[str, Any] = Field(default_factory=dict)
    correlation_metadata: CorrelationMetadata | None = Field(default=None)
    api_version: str = Field(...)

    model_config = ConfigDict(frozen=True, extra="forbid")


class EvaluateClaimResponse(BaseModel):
    """Immutable model representing an evaluation response."""

    claim_id: str = Field(...)
    decision: str = Field(...)
    confidence: float = Field(...)
    explanation: str | None = Field(default=None)

    model_config = ConfigDict(frozen=True, extra="forbid")


class HealthResponse(BaseModel):
    """Immutable model representing health status."""

    status: str = Field(...)
    details: dict[str, str] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True, extra="forbid")


class ReadinessResponse(BaseModel):
    """Immutable model representing readiness status."""

    status: str = Field(...)

    model_config = ConfigDict(frozen=True, extra="forbid")
