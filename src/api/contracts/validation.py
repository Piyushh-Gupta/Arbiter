"""Stateless validators for the API Contract layer."""

import uuid
from typing import Any

from src.api.contracts.versioning import ApiContractProfile, ApiVersionId
from src.core.exceptions import ApiContractValidationError


class PayloadValidator:
    """Validates structural constraints of payloads."""

    def validate_payload_size(
        self, payload: dict[str, Any], max_bytes: int = 1048576
    ) -> None:
        """Validates that a payload does not exceed maximum byte limits."""
        # Simple structural estimation for validation layer
        estimated_size = len(str(payload).encode("utf-8"))
        if estimated_size > max_bytes:
            raise ApiContractValidationError(
                f"Payload exceeds maximum allowed size of {max_bytes} bytes"
            )


class CorrelationIdValidator:
    """Validates correlation ID constraints."""

    def validate(self, correlation_id: str | None, required: bool = True) -> str:
        """Validates the correlation ID format."""
        if not correlation_id:
            if required:
                raise ApiContractValidationError(
                    "Correlation ID is required but was not provided"
                )
            return str(uuid.uuid4())

        # Ensure it's not empty whitespace
        if not correlation_id.strip():
            raise ApiContractValidationError("Correlation ID cannot be empty")

        return correlation_id


class VersionCompatibilityValidator:
    """Validates API version compatibility against active profiles."""

    def validate(
        self, requested_version: str, profile: ApiContractProfile
    ) -> ApiVersionId:
        """Validates that the requested version is supported."""
        try:
            version_id = ApiVersionId(requested_version)
        except ValueError:
            raise ApiContractValidationError(
                f"Invalid API version string: {requested_version}"
            )

        if version_id not in profile.definition.supported_versions:
            raise ApiContractValidationError(
                f"API version {version_id.value} is not supported by profile {profile.profile_id}"
            )

        return version_id


class SchemaCompatibilityValidator:
    """Validates schema structural rules and constraints."""

    def validate_no_extra_fields(
        self, payload: dict[str, Any], allowed_fields: set[str]
    ) -> None:
        """Validates that no extraneous fields exist in the raw payload."""
        provided_fields = set(payload.keys())
        extra_fields = provided_fields - allowed_fields
        if extra_fields:
            raise ApiContractValidationError(
                f"Payload contains unexpected fields: {', '.join(extra_fields)}"
            )
