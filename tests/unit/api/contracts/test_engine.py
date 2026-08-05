"""Unit tests for the ApiContractEngine."""

import pytest

from src.api.contracts.engine import ApiContractEngine
from src.api.contracts.versioning import (
    ApiContractDefinition,
    ApiContractProfile,
    ApiContractRegistry,
    ApiVersionId,
)
from src.core.exceptions import ApiContractValidationError


@pytest.fixture
def contract_engine() -> ApiContractEngine:
    """Fixture for a configured ApiContractEngine."""
    definition = ApiContractDefinition(
        supported_versions=(ApiVersionId.V1,),
        require_correlation_id=True,
        strict_validation=True,
    )
    profile = ApiContractProfile(
        profile_id="test_profile",
        definition=definition,
    )
    registry = ApiContractRegistry(profiles=[profile])
    return ApiContractEngine(registry=registry, active_profile_id="test_profile")


def test_engine_initialization(contract_engine: ApiContractEngine) -> None:
    """Verifies the engine initializes correctly."""
    assert contract_engine.active_profile_id == "test_profile"


def test_engine_validate_request_structure(contract_engine: ApiContractEngine) -> None:
    """Verifies request structure validation via the engine."""
    payload = {"claim": "Test claim"}

    version, correlation_id = contract_engine.validate_request_structure(
        payload=payload, requested_version="v1", correlation_id="corr-123"
    )

    assert version == ApiVersionId.V1
    assert correlation_id == "corr-123"


def test_engine_validate_request_structure_missing_correlation(
    contract_engine: ApiContractEngine,
) -> None:
    """Verifies engine rejects missing correlation ID when required."""
    payload = {"claim": "Test claim"}

    with pytest.raises(ApiContractValidationError):
        contract_engine.validate_request_structure(
            payload=payload, requested_version="v1", correlation_id=None
        )


def test_engine_validate_request_structure_invalid_version(
    contract_engine: ApiContractEngine,
) -> None:
    """Verifies engine rejects unsupported API versions."""
    payload = {"claim": "Test claim"}

    with pytest.raises(ApiContractValidationError):
        contract_engine.validate_request_structure(
            payload=payload, requested_version="v2", correlation_id="corr-123"
        )


def test_engine_build_success_response(contract_engine: ApiContractEngine) -> None:
    """Verifies building a successful response envelope."""
    response = contract_engine.build_success_response(
        data={"result": "success"},
        api_version=ApiVersionId.V1,
        correlation_id="corr-123",
    )

    assert response.data == {"result": "success"}
    assert response.api_version == "v1"
    assert response.correlation_metadata is not None
    assert response.correlation_metadata.correlation_id == "corr-123"


def test_engine_build_error_response(contract_engine: ApiContractEngine) -> None:
    """Verifies building an error response envelope."""
    error = contract_engine.build_error_response(
        error_code="TEST_ERROR",
        message="A test error occurred",
        correlation_id="corr-123",
    )

    assert error.error_code == "TEST_ERROR"
    assert error.message == "A test error occurred"
    assert error.correlation_id == "corr-123"
    assert error.details is None
