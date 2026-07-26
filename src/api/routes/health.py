"""Health check endpoint."""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    """Transport model for the health check response."""

    status: str


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Returns the status of the Arbiter API.",
)
async def health_check() -> HealthResponse:
    """Returns application health status."""
    return HealthResponse(status="ok")
