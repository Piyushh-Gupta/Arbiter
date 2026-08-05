"""Unit tests for the API Contract validators."""

import pytest

from src.api.contracts.validation import (
    CorrelationIdValidator,
    PayloadValidator,
    SchemaCompatibilityValidator,
    VersionCompatibilityValidator,
)
from src.api.contracts.versioning import (
    ApiContractDefinition,
    ApiContractProfile,
    ApiVersionId,
)
from src.core.exceptions import ApiContractValidationError


def test_correlation_id_validator() -> None:
    """Verifies correlation ID validation logic."""
    validator = CorrelationIdValidator()

    # Valid correlation ID
    assert validator.validate("test-123") == "test-123"

    # Missing but not required (generates new)
    generated = validator.validate(None, required=False)
    assert generated is not None
    assert isinstance(generated, str)

    # Missing and required
    with pytest.raises(ApiContractValidationError):
        validator.validate(None, required=True)

    # Empty string
    with pytest.raises(ApiContractValidationError):
        validator.validate("   ", required=True)


def test_payload_validator() -> None:
    """Verifies payload size validation."""
    validator = PayloadValidator()

    # Valid size
    validator.validate_payload_size({"key": "value"}, max_bytes=100)

    # Exceeds max bytes
    with pytest.raises(ApiContractValidationError):
        validator.validate_payload_size({"key": "value" * 100}, max_bytes=10)


def test_version_compatibility_validator() -> None:
    """Verifies API version compatibility validation."""
    validator = VersionCompatibilityValidator()
    profile = ApiContractProfile(
        profile_id="test",
        definition=ApiContractDefinition(supported_versions=(ApiVersionId.V1,)),
    )

    # Supported version
    assert validator.validate("v1", profile) == ApiVersionId.V1

    # Unsupported version
    with pytest.raises(ApiContractValidationError):
        validator.validate("v2", profile)

    # Invalid version string
    with pytest.raises(ApiContractValidationError):
        validator.validate("v99", profile)


def test_schema_compatibility_validator() -> None:
    """Verifies schema strictness constraints."""
    validator = SchemaCompatibilityValidator()
    allowed = {"valid_key", "another_key"}

    # Valid payload
    validator.validate_no_extra_fields({"valid_key": 1}, allowed)

    # Invalid payload
    with pytest.raises(ApiContractValidationError):
        validator.validate_no_extra_fields({"invalid_key": 1}, allowed)
