"""Unit tests for service models."""

import pytest
from pydantic import ValidationError

from src.api.services.service_models import (
    ClientMetadata,
    EvaluationRequest,
    RequestMetadata,
    ServiceContext,
)


def test_request_metadata_frozen() -> None:
    rm = RequestMetadata(
        headers={"x-foo": "bar"}, query_params={"q": "1"}, client_ip="127.0.0.1"
    )
    with pytest.raises(ValidationError):
        rm.client_ip = "192.168.1.1"


def test_service_context_initialization() -> None:
    ctx = ServiceContext(
        correlation_id="1234",
        request_metadata=RequestMetadata(),
        client_metadata=ClientMetadata(),
    )
    assert ctx.correlation_id == "1234"
    assert ctx.request_metadata.client_ip is None


def test_evaluation_request_validation() -> None:
    with pytest.raises(ValidationError):
        EvaluationRequest(  # type: ignore
            claim="test", pipeline_profile_id="default"
        )  # missing context
