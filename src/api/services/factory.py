"""Service factory."""

from src.api.services.base import BaseEvaluationService, BaseHealthService
from src.api.services.evaluation_service import EvaluationService
from src.api.services.health_service import HealthService
from src.api.services.registry import ServiceRegistry
from src.core.pipeline.orchestrator import ArbiterPipeline


class ServiceFactory:
    """Factory for instantiating stateless services."""

    @staticmethod
    def build_evaluation_service(pipeline: ArbiterPipeline) -> BaseEvaluationService:
        """Builds the evaluation service."""
        return EvaluationService(pipeline=pipeline)

    @staticmethod
    def build_health_service(pipeline: ArbiterPipeline) -> BaseHealthService:
        """Builds the health service."""
        return HealthService(pipeline=pipeline)

    @classmethod
    def build_registry(cls, pipeline: ArbiterPipeline) -> ServiceRegistry:
        """Builds the global service registry."""
        return ServiceRegistry(
            evaluation_service=cls.build_evaluation_service(pipeline),
            health_service=cls.build_health_service(pipeline),
        )
