"""Health check endpoints."""

from fastapi import APIRouter, Request, Response, status
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    """Transport model for the health check response."""

    status: str


@router.get(
    "/health/live",
    response_model=HealthResponse,
    summary="Liveness Check",
    description="Returns HTTP 200 indicating the application process and event loop are alive.",
)
async def liveness_check() -> HealthResponse:
    """Returns application liveness status."""
    return HealthResponse(status="alive")


@router.get(
    "/health/ready",
    response_model=HealthResponse,
    summary="Readiness Check",
    description="Returns HTTP 200 if startup completed successfully and the pipeline is mounted. Otherwise 503.",
)
async def readiness_check(request: Request, response: Response) -> HealthResponse:
    """Returns application readiness status."""
    if (
        hasattr(request.app.state, "pipeline")
        and request.app.state.pipeline is not None
    ):
        return HealthResponse(status="ready")

    response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(status="not_ready")


@router.get(
    "/verification/health/live",
    response_model=HealthResponse,
    summary="Verification Subsystem Liveness Check",
    description="Returns HTTP 200 indicating the verification subsystem is alive.",
)
async def verification_liveness_check() -> HealthResponse:
    """Returns verification subsystem liveness status."""
    return HealthResponse(status="alive")


@router.get(
    "/verification/health/ready",
    response_model=HealthResponse,
    summary="Verification Subsystem Readiness Check",
    description="Returns HTTP 200 if all verification-related registries are loaded. Otherwise 503.",
)
async def verification_readiness_check(
    request: Request, response: Response
) -> HealthResponse:
    """Returns verification subsystem readiness status."""
    app_state = request.app.state
    pipeline = getattr(app_state, "pipeline", None)
    opt_registry = getattr(app_state, "verification_optimization_registry", None)
    cal_registry = getattr(app_state, "calibration_registry", None)

    if pipeline is not None and opt_registry is not None and cal_registry is not None:
        ver_reg = getattr(pipeline, "_verification_registry", None)
        exp_reg = getattr(pipeline, "_explanation_registry", None)
        if ver_reg is not None and exp_reg is not None:
            return HealthResponse(status="ready")

    response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(status="not_ready")
