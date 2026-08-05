"""Service layer protocols."""

from typing import Protocol, runtime_checkable

from src.api.services.service_models import (
    EvaluationRequest,
    EvaluationResponse,
    HealthStatusResponse,
    ServiceContext,
)


@runtime_checkable
class BaseEvaluationService(Protocol):
    """Protocol for evaluation service operations."""

    def evaluate(self, request: EvaluationRequest) -> EvaluationResponse:
        """Executes a claim evaluation."""
        ...


@runtime_checkable
class BaseHealthService(Protocol):
    """Protocol for health service operations."""

    def check_liveness(self, context: ServiceContext) -> HealthStatusResponse:
        """Checks liveness of the application."""
        ...

    def check_readiness(self, context: ServiceContext) -> HealthStatusResponse:
        """Checks readiness of the application."""
        ...
