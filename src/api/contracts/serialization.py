"""Serialization utilities for the API Contract layer."""

from typing import Any, Mapping, TypeVar

from src.api.contracts.error_models import ErrorEnvelope, ValidationErrorDetail
from src.api.contracts.response_models import ApiResponseEnvelope, CorrelationMetadata
from src.api.contracts.versioning import ApiVersionId

T = TypeVar("T")


class ResponseSerializer:
    """Stateless serializer for standardized response envelopes."""

    def serialize_success(
        self,
        data: T,
        api_version: ApiVersionId,
        correlation_id: str,
        client_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ApiResponseEnvelope[T]:
        """Serializes successful response payload into a standardized envelope."""
        correlation_meta = CorrelationMetadata(
            correlation_id=correlation_id,
            client_id=client_id,
        )
        return ApiResponseEnvelope(
            data=data,
            api_version=api_version.value,
            correlation_metadata=correlation_meta,
            metadata=dict(metadata) if metadata else {},
        )

    def serialize_error(
        self,
        error_code: str,
        message: str,
        correlation_id: str | None = None,
        details: list[ValidationErrorDetail] | Mapping[str, Any] | None = None,
    ) -> ErrorEnvelope:
        """Serializes an error response into a standardized error envelope."""
        return ErrorEnvelope(
            error_code=error_code,
            message=message,
            correlation_id=correlation_id,
            details=details,
        )
