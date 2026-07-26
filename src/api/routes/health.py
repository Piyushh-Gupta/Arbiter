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
