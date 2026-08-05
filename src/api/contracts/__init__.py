"""Public exports for the API Contract layer."""

from src.api.contracts.engine import ApiContractEngine
from src.api.contracts.error_models import ErrorEnvelope, ValidationErrorDetail
from src.api.contracts.request_models import EvaluateClaimRequest, PaginationMetadata
from src.api.contracts.response_models import (
    ApiResponseEnvelope,
    CorrelationMetadata,
    EvaluateClaimResponse,
    HealthResponse,
    ReadinessResponse,
)
from src.api.contracts.serialization import ResponseSerializer
from src.api.contracts.validation import (
    CorrelationIdValidator,
    PayloadValidator,
    SchemaCompatibilityValidator,
    VersionCompatibilityValidator,
)
from src.api.contracts.versioning import (
    ApiContractDefinition,
    ApiContractProfile,
    ApiContractRegistry,
    ApiVersionId,
)

__all__ = [
    "ApiContractEngine",
    "ErrorEnvelope",
    "ValidationErrorDetail",
    "EvaluateClaimRequest",
    "PaginationMetadata",
    "ApiResponseEnvelope",
    "CorrelationMetadata",
    "EvaluateClaimResponse",
    "HealthResponse",
    "ReadinessResponse",
    "ResponseSerializer",
    "CorrelationIdValidator",
    "PayloadValidator",
    "SchemaCompatibilityValidator",
    "VersionCompatibilityValidator",
    "ApiContractDefinition",
    "ApiContractProfile",
    "ApiContractRegistry",
    "ApiVersionId",
]
