import os

os.makedirs("src/api/services", exist_ok=True)

with open("src/api/services/__init__.py", "w", encoding="utf-8") as f:
    f.write('"""Service layer module."""\n')

with open("src/api/services/service_models.py", "w", encoding="utf-8") as f:
    f.write('''"""Service layer immutable models."""

from typing import Mapping
from pydantic import BaseModel, ConfigDict, Field


class RequestMetadata(BaseModel):
    """Immutable request metadata."""
    model_config = ConfigDict(frozen=True)
    headers: Mapping[str, str] = Field(default_factory=dict)
    query_params: Mapping[str, str] = Field(default_factory=dict)
    client_ip: str | None = None


class ClientMetadata(BaseModel):
    """Immutable client metadata."""
    model_config = ConfigDict(frozen=True)
    user_agent: str | None = None
    client_id: str | None = None


class ServiceExecutionMetadata(BaseModel):
    """Immutable service execution metadata."""
    model_config = ConfigDict(frozen=True)
    start_time_ns: int
    end_time_ns: int
    duration_ms: float


class ServiceContext(BaseModel):
    """Immutable service context containing correlation ID."""
    model_config = ConfigDict(frozen=True)
    correlation_id: str
    request_metadata: RequestMetadata = Field(default_factory=RequestMetadata)
    client_metadata: ClientMetadata = Field(default_factory=ClientMetadata)


class EvaluationRequest(BaseModel):
    """Immutable domain evaluation request."""
    model_config = ConfigDict(frozen=True)
    claim: str
    pipeline_profile_id: str
    context: ServiceContext


class MetricServiceDTO(BaseModel):
    """Immutable service metric representation."""
    model_config = ConfigDict(frozen=True)
    identifier: str
    title: str
    score: float


class EvaluationResponse(BaseModel):
    """Immutable domain evaluation response."""
    model_config = ConfigDict(frozen=True)
    metrics: tuple[MetricServiceDTO, ...]
    execution_metadata: ServiceExecutionMetadata
    correlation_id: str


class HealthStatusResponse(BaseModel):
    """Immutable health status response."""
    model_config = ConfigDict(frozen=True)
    status: str
    correlation_id: str
''')

with open("src/api/services/profiles.py", "w", encoding="utf-8") as f:
    f.write('''"""Service layer configuration profiles."""

from pydantic import BaseModel, ConfigDict


class ServiceProfile(BaseModel):
    """Immutable service configuration profile."""
    model_config = ConfigDict(frozen=True)
    profile_id: str
    require_correlation_id: bool = True
    timeout_seconds: float = 30.0
''')

with open("src/api/services/base.py", "w", encoding="utf-8") as f:
    f.write('''"""Service layer protocols."""

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
''')

with open("src/api/services/registry.py", "w", encoding="utf-8") as f:
    f.write('''"""Service layer registries."""

from src.api.services.base import BaseEvaluationService, BaseHealthService
from src.api.services.profiles import ServiceProfile
from src.core.exceptions import ArbiterError


class ServiceProfileNotFoundError(ArbiterError):
    """Raised when a service profile cannot be found."""


class DuplicateServiceProfileError(ArbiterError):
    """Raised when registering a duplicate service profile."""


class ServiceProfileRegistry:
    """Registry for service profiles."""

    def __init__(self, profiles: tuple[ServiceProfile, ...] = ()) -> None:
        self._profiles: dict[str, ServiceProfile] = {}
        for p in profiles:
            self.register(p)

    def register(self, profile: ServiceProfile) -> None:
        """Registers a service profile."""
        if profile.profile_id in self._profiles:
            raise DuplicateServiceProfileError(f"Duplicate profile: {profile.profile_id}")
        self._profiles[profile.profile_id] = profile

    def resolve(self, profile_id: str) -> ServiceProfile:
        """Resolves a service profile by ID."""
        if profile_id not in self._profiles:
            raise ServiceProfileNotFoundError(f"Profile not found: {profile_id}")
        return self._profiles[profile_id]


class ServiceRegistry:
    """Registry holding instantiated services."""

    def __init__(
        self,
        evaluation_service: BaseEvaluationService,
        health_service: BaseHealthService,
    ) -> None:
        self._evaluation_service = evaluation_service
        self._health_service = health_service

    @property
    def evaluation_service(self) -> BaseEvaluationService:
        """Returns the evaluation service."""
        return self._evaluation_service

    @property
    def health_service(self) -> BaseHealthService:
        """Returns the health service."""
        return self._health_service
''')

with open("src/api/services/exceptions.py", "w", encoding="utf-8") as f:
    f.write('''"""Exception translation layer."""

from fastapi import HTTPException, status
from pydantic import ValidationError

from src.core.exceptions import ArbiterError, ConfigurationError


class ExceptionTranslator:
    """Translates domain exceptions to HTTP exceptions deterministically."""

    @staticmethod
    def translate(exc: Exception) -> HTTPException:
        """Translates an exception to an HTTPException."""
        if isinstance(exc, ValidationError):
            return HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            )
        if isinstance(exc, ConfigurationError):
            return HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            )
        if isinstance(exc, ArbiterError):
            return HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            )
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error",
        )
''')

with open("src/api/services/evaluation_service.py", "w", encoding="utf-8") as f:
    f.write('''"""Evaluation service implementation."""

import time

from src.api.services.base import BaseEvaluationService
from src.api.services.service_models import (
    EvaluationRequest,
    EvaluationResponse,
    MetricServiceDTO,
    ServiceExecutionMetadata,
)
from src.core.pipeline.orchestrator import ArbiterPipeline
from src.core.pipeline.pipeline_models import PipelineExecutionRequest


class EvaluationService(BaseEvaluationService):
    """Stateless evaluation service."""

    def __init__(self, pipeline: ArbiterPipeline) -> None:
        self._pipeline = pipeline

    def evaluate(self, request: EvaluationRequest) -> EvaluationResponse:
        """Executes the evaluation request deterministically."""
        start_ns = time.perf_counter_ns()
        
        domain_req = PipelineExecutionRequest(
            claim=request.claim,
            pipeline_profile_id=request.pipeline_profile_id,
        )
        
        domain_res = self._pipeline.execute(domain_req)
        
        metrics = tuple(
            MetricServiceDTO(identifier=m.identifier, title=m.title, score=m.score)
            for m in domain_res.metrics
        )
        
        end_ns = time.perf_counter_ns()
        duration_ms = (end_ns - start_ns) / 1_000_000.0
        
        return EvaluationResponse(
            metrics=metrics,
            execution_metadata=ServiceExecutionMetadata(
                start_time_ns=start_ns,
                end_time_ns=end_ns,
                duration_ms=duration_ms,
            ),
            correlation_id=request.context.correlation_id,
        )
''')

with open("src/api/services/health_service.py", "w", encoding="utf-8") as f:
    f.write('''"""Health service implementation."""

from src.api.services.base import BaseHealthService
from src.api.services.service_models import (
    HealthStatusResponse,
    ServiceContext,
)
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
            if snapshot.readiness_status == PipelineReadinessStatus.READY:
                return HealthStatusResponse(
                    status="ready",
                    correlation_id=context.correlation_id,
                )
        return HealthStatusResponse(
            status="not_ready",
            correlation_id=context.correlation_id,
        )
''')

with open("src/api/services/factory.py", "w", encoding="utf-8") as f:
    f.write('''"""Service factory."""

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
''')
