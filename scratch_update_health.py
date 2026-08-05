with open("src/api/routes/health.py", "w", encoding="utf-8") as f:
    f.write('''"""Health check endpoints."""

import uuid

from fastapi import APIRouter, Request, Response, status

from src.api.services.registry import ServiceRegistry
from src.api.services.service_models import (
    ClientMetadata,
    HealthStatusResponse,
    RequestMetadata,
    ServiceContext,
)

router = APIRouter()


def _build_context(request: Request) -> ServiceContext:
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    return ServiceContext(
        correlation_id=correlation_id,
        request_metadata=RequestMetadata(
            headers=dict(request.headers),
            query_params=dict(request.query_params),
            client_ip=request.client.host if request.client else None,
        ),
        client_metadata=ClientMetadata(
            user_agent=request.headers.get("user-agent"),
        ),
    )


@router.get(
    "/health/live",
    response_model=HealthStatusResponse,
    summary="Liveness Check",
    description="Returns HTTP 200 indicating the application process and event loop are alive.",
)
async def liveness_check(request: Request) -> HealthStatusResponse:
    """Returns application liveness status through the Service Layer."""
    registry: ServiceRegistry = request.app.state.service_registry
    return registry.health_service.check_liveness(_build_context(request))


@router.get(
    "/health/ready",
    response_model=HealthStatusResponse,
    summary="Readiness Check",
    description="Returns HTTP 200 if startup completed successfully and the pipeline is mounted. Otherwise 503.",
)
async def readiness_check(request: Request, response: Response) -> HealthStatusResponse:
    """Returns application readiness status through the Service Layer."""
    if not hasattr(request.app.state, "service_registry") or request.app.state.service_registry is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthStatusResponse(status="not_ready", correlation_id="")
        
    registry: ServiceRegistry = request.app.state.service_registry
    res = registry.health_service.check_readiness(_build_context(request))
    if res.status != "ready":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return res


@router.get(
    "/verification/health/live",
    response_model=HealthStatusResponse,
    summary="Verification Subsystem Liveness Check",
    description="Returns HTTP 200 indicating the verification subsystem is alive.",
)
async def verification_liveness_check(request: Request) -> HealthStatusResponse:
    """Returns verification subsystem liveness status."""
    registry: ServiceRegistry = request.app.state.service_registry
    return registry.health_service.check_liveness(_build_context(request))


@router.get(
    "/verification/health/ready",
    response_model=HealthStatusResponse,
    summary="Verification Subsystem Readiness Check",
    description="Returns HTTP 200 if all verification-related registries are loaded. Otherwise 503.",
)
async def verification_readiness_check(
    request: Request, response: Response
) -> HealthStatusResponse:
    """Returns verification subsystem readiness status."""
    if not hasattr(request.app.state, "service_registry") or request.app.state.service_registry is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthStatusResponse(status="not_ready", correlation_id="")
        
    registry: ServiceRegistry = request.app.state.service_registry
    res = registry.health_service.check_readiness(_build_context(request))
    if res.status != "ready":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return res
''')
