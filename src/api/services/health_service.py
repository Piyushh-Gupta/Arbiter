"""Health service implementation."""

from src.api.services.base import BaseHealthService
from src.api.services.service_models import HealthStatusResponse, ServiceContext
from src.core.pipeline.operations.operation_models import PipelineReadinessStatus
from src.core.pipeline.orchestrator import ArbiterPipeline


class HealthService(BaseHealthService):
    """Stateless health service."""

    def __init__(self, pipeline: ArbiterPipeline) -> None:
        self._pipeline = pipeline

    def check_liveness(self, context: ServiceContext) -> HealthStatusResponse:
        """Checks liveness of the application."""
        return HealthStatusResponse(
            status="alive",
            correlation_id=context.correlation_id,
        )

    def check_readiness(self, context: ServiceContext) -> HealthStatusResponse:
        """Checks readiness of the application."""
        if hasattr(self._pipeline, "operations"):
            snapshot = self._pipeline.operations.get_snapshot()
            if snapshot.overall_readiness == PipelineReadinessStatus.READY:
                return HealthStatusResponse(
                    status="ready",
                    correlation_id=context.correlation_id,
                )
        return HealthStatusResponse(
            status="not_ready",
            correlation_id=context.correlation_id,
        )
