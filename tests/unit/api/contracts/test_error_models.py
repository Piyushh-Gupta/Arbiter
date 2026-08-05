"""Unit tests for the API Contract error models."""

import pytest
from pydantic import ValidationError

from src.api.contracts.error_models import ErrorEnvelope, ValidationErrorDetail


def test_validation_error_detail() -> None:
    """Verifies ValidationErrorDetail structure."""
    detail = ValidationErrorDetail(
        loc=("body", "claim"), msg="Field required", type="value_error.missing"
    )
    assert detail.loc == ("body", "claim")
    assert detail.msg == "Field required"
    assert detail.type == "value_error.missing"


def test_error_envelope() -> None:
    """Verifies ErrorEnvelope structure."""
    detail = ValidationErrorDetail(
        loc=("body", "claim"), msg="Field required", type="value_error.missing"
    )
    envelope = ErrorEnvelope(
        error_code="VALIDATION_ERROR",
        message="Request validation failed",
        correlation_id="corr-123",
        details=[detail],
    )

    assert envelope.error_code == "VALIDATION_ERROR"
    assert envelope.message == "Request validation failed"
    assert envelope.correlation_id == "corr-123"
    assert envelope.details is not None
    assert len(envelope.details) == 1
    assert envelope.details[0].loc == ("body", "claim")  # type: ignore


def test_error_envelope_immutability() -> None:
    """Verifies ErrorEnvelope is immutable."""
    envelope = ErrorEnvelope(
        error_code="TEST",
        message="Test message",
    )
    with pytest.raises(ValidationError):
        envelope.error_code = "MODIFIED"
