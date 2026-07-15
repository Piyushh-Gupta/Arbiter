from fastapi import FastAPI
from contextlib import asynccontextmanager
import structlog

from src.core.logging import setup_logging
from src.core.config import settings

logger = structlog.get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(settings.log_level)
    logger.info("Starting up Arbiter API", env=settings.environment)
    yield
    logger.info("Shutting down Arbiter API")

app = FastAPI(
    title=settings.app_name,
    description="Trustworthy AI Decision Support System for Evidence-Based Claim Verification",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    return {"status": "ok"}
