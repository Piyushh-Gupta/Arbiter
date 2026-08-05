"""Unit tests for API contract response models."""

import pytest
from pydantic import ValidationError

from src.api.contracts.response_models import (
    ApiResponseEnvelope,
    CorrelationMetadata,
    EvaluateClaimResponse,
    HealthResponse,
    ReadinessResponse,
)


def test_correlation_metadata() -> None:
    """Verifies correlation metadata instantiation."""
    model = CorrelationMetadata(correlation_id="test-id")
    assert model.correlation_id == "test-id"
    assert model.client_id is None

    with pytest.raises(ValidationError):
        CorrelationMetadata(correlation_id="test-id", extra="field")  # type: ignore 


def test_api_response_envelope() -> None:
    """Verifies generic API response envelope."""
    data = EvaluateClaimResponse(claim_id="123", decision="SUPPORTED", confidence=0.9)
    meta = CorrelationMetadata(correlation_id="corr-1")

    envelope = ApiResponseEnvelope[EvaluateClaimResponse](
        data=data,
        api_version="v1",
        correlation_metadata=meta,
    )

    assert envelope.data.claim_id == "123"
    assert envelope.api_version == "v1"
    assert envelope.correlation_metadata is not None
    assert envelope.correlation_metadata.correlation_id == "corr-1"


def test_health_and_readiness_responses() -> None:
    """Verifies health and readiness response models."""
    health = HealthResponse(status="alive", details={"db": "up"})
    assert health.status == "alive"
    assert health.details["db"] == "up"

    ready = ReadinessResponse(status="ready")
    assert ready.status == "ready"


def test_response_models_immutability() -> None:
    """Verifies response models are immutable."""
    model = ReadinessResponse(status="ready")
    with pytest.raises(ValidationError):
        model.status = "not_ready" 
