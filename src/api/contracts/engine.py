"""ApiContractEngine implementation for the API Contract layer."""

from typing import Any, Mapping, TypeVar

from src.api.contracts.error_models import ErrorEnvelope, ValidationErrorDetail
from src.api.contracts.response_models import ApiResponseEnvelope
from src.api.contracts.serialization import ResponseSerializer
from src.api.contracts.validation import (
    CorrelationIdValidator,
    PayloadValidator,
    SchemaCompatibilityValidator,
    VersionCompatibilityValidator,
)
from src.api.contracts.versioning import ApiContractRegistry, ApiVersionId

T = TypeVar("T")


class ApiContractEngine:
    """Orchestration engine for the API contracts subsystem.

    This engine acts as the single entrypoint for structural validation
    and response serialization, ensuring strict separation from business logic.
    """

    def __init__(
        self,
        registry: ApiContractRegistry,
        active_profile_id: str,
        payload_validator: PayloadValidator | None = None,
        correlation_validator: CorrelationIdValidator | None = None,
        version_validator: VersionCompatibilityValidator | None = None,
        schema_validator: SchemaCompatibilityValidator | None = None,
        serializer: ResponseSerializer | None = None,
    ) -> None:
        """Initializes the ApiContractEngine with required dependencies."""
        self._registry = registry
        self._profile = self._registry.resolve(active_profile_id)

        # Instantiate default components if none provided
        self._payload_validator = payload_validator or PayloadValidator()
        self._correlation_validator = correlation_validator or CorrelationIdValidator()
        self._version_validator = version_validator or VersionCompatibilityValidator()
        self._schema_validator = schema_validator or SchemaCompatibilityValidator()
        self._serializer = serializer or ResponseSerializer()

    @property
    def active_profile_id(self) -> str:
        """Returns the active contract profile ID."""
        return self._profile.profile_id

    def validate_request_structure(
        self,
        payload: dict[str, Any],
        requested_version: str,
        correlation_id: str | None = None,
    ) -> tuple[ApiVersionId, str]:
        """Validates incoming structural boundaries before processing.

        Returns:
            Tuple containing the resolved ApiVersionId and the valid correlation ID.

        Raises:
            ApiContractValidationError: If any structural validation fails.
        """
        version_id = self._version_validator.validate(requested_version, self._profile)
        valid_correlation_id = self._correlation_validator.validate(
            correlation_id, required=self._profile.definition.require_correlation_id
        )

        if self._profile.definition.strict_validation:
            self._payload_validator.validate_payload_size(payload)

        return version_id, valid_correlation_id

    def build_success_response(
        self,
        data: T,
        api_version: ApiVersionId,
        correlation_id: str,
        client_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ApiResponseEnvelope[T]:
        """Serializes successful response data into a standard envelope."""
        return self._serializer.serialize_success(
            data=data,
            api_version=api_version,
            correlation_id=correlation_id,
            client_id=client_id,
            metadata=metadata,
        )

    def build_error_response(
        self,
        error_code: str,
        message: str,
        correlation_id: str | None = None,
        details: list[ValidationErrorDetail] | Mapping[str, Any] | None = None,
    ) -> ErrorEnvelope:
        """Serializes failure information into a standard error envelope."""
        return self._serializer.serialize_error(
            error_code=error_code,
            message=message,
            correlation_id=correlation_id,
            details=details,
        )
