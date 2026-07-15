from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from src.core.bootstrap import initialize_application
from src.core.config import settings
from src.core.logging import setup_logging

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan events for the application."""
    setup_logging(settings.log.level)
    initialize_application()
    logger.info("Starting up Arbiter API", env=settings.environment)
    yield
    logger.info("Shutting down Arbiter API")


app = FastAPI(
    title=settings.app_name,
    description="Trustworthy AI Decision Support System for Evidence-Based Claim Verification",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}
