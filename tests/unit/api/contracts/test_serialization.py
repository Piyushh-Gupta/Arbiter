"""Unit tests for the API Contract serialization layer."""

from src.api.contracts.serialization import ResponseSerializer
from src.api.contracts.versioning import ApiVersionId


def test_serialize_success() -> None:
    """Verifies successful response serialization."""
    serializer = ResponseSerializer()

    response = serializer.serialize_success(
        data={"key": "value"},
        api_version=ApiVersionId.V1,
        correlation_id="corr-123",
        client_id="client-456",
        metadata={"time": 100},
    )

    assert response.data == {"key": "value"}
    assert response.api_version == "v1"
    assert response.correlation_metadata is not None
    assert response.correlation_metadata.correlation_id == "corr-123"
    assert response.correlation_metadata.client_id == "client-456"
    assert response.metadata == {"time": 100}


def test_serialize_error() -> None:
    """Verifies error response serialization."""
    serializer = ResponseSerializer()

    error = serializer.serialize_error(
        error_code="TEST_ERR",
        message="Test message",
        correlation_id="corr-123",
        details={"field": "invalid"},
    )

    assert error.error_code == "TEST_ERR"
    assert error.message == "Test message"
    assert error.correlation_id == "corr-123"
    assert error.details == {"field": "invalid"}
