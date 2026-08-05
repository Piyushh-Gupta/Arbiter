"""Immutable models for the API middleware and request lifecycle."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RequestLifecyclePhase(str, Enum):
    """Strongly typed lifecycle phases for request processing."""

    REQUEST_RECEIVED = "REQUEST_RECEIVED"
    CORRELATION_ESTABLISHED = "CORRELATION_ESTABLISHED"
    VALIDATED = "VALIDATED"
    SERVICE_EXECUTION = "SERVICE_EXECUTION"
    RESPONSE_GENERATED = "RESPONSE_GENERATED"
    FINALIZED = "FINALIZED"


class CorrelationContext(BaseModel):
    """Immutable model representing correlation context for a request."""

    correlation_id: str = Field(...)
    client_id: str | None = Field(default=None)

    model_config = ConfigDict(frozen=True, extra="forbid")


class RequestTiming(BaseModel):
    """Immutable model capturing request timing metrics."""

    start_time_ns: int = Field(default=0)
    end_time_ns: int = Field(default=0)

    @property
    def elapsed_ms(self) -> float:
        """Returns the elapsed time in milliseconds."""
        if self.start_time_ns == 0 or self.end_time_ns == 0:
            return 0.0
        return (self.end_time_ns - self.start_time_ns) / 1_000_000.0

    model_config = ConfigDict(frozen=True, extra="forbid")


class MiddlewareExecutionContext(BaseModel):
    """Immutable state passed between middleware components."""

    request: Any = Field(..., description="The raw transport request")
    correlation_context: CorrelationContext | None = Field(default=None)
    timing: RequestTiming = Field(default_factory=RequestTiming)
    contract_profile_id: str | None = Field(default=None)
    service_profile_id: str | None = Field(default=None)
    phase: RequestLifecyclePhase = Field(default=RequestLifecyclePhase.REQUEST_RECEIVED)

    model_config = ConfigDict(frozen=True, extra="forbid")


class MiddlewareProfile(BaseModel):
    """Immutable profile defining middleware policies."""

    profile_id: str = Field(...)
    require_correlation_propagation: bool = Field(default=True)

    model_config = ConfigDict(frozen=True, extra="forbid")
