"""Service layer immutable models."""

from typing import Mapping

from pydantic import BaseModel, ConfigDict, Field


class RequestMetadata(BaseModel):
    """Immutable request metadata."""

    model_config = ConfigDict(frozen=True)
    headers: Mapping[str, str] = Field(default_factory=dict)
    query_params: Mapping[str, str] = Field(default_factory=dict)
    client_ip: str | None = None


class ClientMetadata(BaseModel):
    """Immutable client metadata."""

    model_config = ConfigDict(frozen=True)
    user_agent: str | None = None
    client_id: str | None = None


class ServiceExecutionMetadata(BaseModel):
    """Immutable service execution metadata."""

    model_config = ConfigDict(frozen=True)
    start_time_ns: int
    end_time_ns: int
    duration_ms: float


class ServiceContext(BaseModel):
    """Immutable service context containing correlation ID."""

    model_config = ConfigDict(frozen=True)
    correlation_id: str
    request_metadata: RequestMetadata = Field(default_factory=RequestMetadata)
    client_metadata: ClientMetadata = Field(default_factory=ClientMetadata)


class EvaluationRequest(BaseModel):
    """Immutable domain evaluation request."""

    model_config = ConfigDict(frozen=True)
    claim: str
    pipeline_profile_id: str
    context: ServiceContext


class MetricServiceDTO(BaseModel):
    """Immutable service metric representation."""

    model_config = ConfigDict(frozen=True)
    identifier: str
    title: str
    score: float


class EvaluationResponse(BaseModel):
    """Immutable domain evaluation response."""

    model_config = ConfigDict(frozen=True)
    metrics: tuple[MetricServiceDTO, ...]
    execution_metadata: ServiceExecutionMetadata
    correlation_id: str


class HealthStatusResponse(BaseModel):
    """Immutable health status response."""

    model_config = ConfigDict(frozen=True)
    status: str
    correlation_id: str
