"""Unit tests for API contract request models."""

import pytest
from pydantic import ValidationError

from src.api.contracts.request_models import EvaluateClaimRequest, PaginationMetadata


def test_pagination_metadata_defaults() -> None:
    """Verifies pagination metadata defaults."""
    model = PaginationMetadata()
    assert model.page == 1
    assert model.page_size == 50


def test_pagination_metadata_constraints() -> None:
    """Verifies pagination constraints."""
    with pytest.raises(ValidationError):
        PaginationMetadata(page=0)
    with pytest.raises(ValidationError):
        PaginationMetadata(page_size=101)


def test_evaluate_claim_request_validation() -> None:
    """Verifies EvaluateClaimRequest validation constraints."""
    model = EvaluateClaimRequest(claim="Test claim")
    assert model.claim == "Test claim"
    assert model.context is None
    assert model.metadata == {}

    with pytest.raises(ValidationError):
        EvaluateClaimRequest(claim="")

    with pytest.raises(ValidationError):
        # Extra fields forbidden
        EvaluateClaimRequest(claim="Test", extra="field")  # type: ignore 


def test_request_models_immutability() -> None:
    """Verifies request models are immutable."""
    model = EvaluateClaimRequest(claim="Test")
    with pytest.raises(ValidationError):
        model.claim = "Modified" 
