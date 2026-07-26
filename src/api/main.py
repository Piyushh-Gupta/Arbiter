import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from src.api.routes import evaluation, health
from src.core.bootstrap import build_pipeline, initialize_application
from src.core.config import Settings
from src.core.exceptions import ArbiterError

logger = logging.getLogger("arbiter.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lifespan events for the FastAPI application."""
    config = Settings()
    try:
        initialize_application(config)
        pipeline = build_pipeline(config)
        app.state.pipeline = pipeline
        logger.info("Arbiter Pipeline mounted successfully.")
    except Exception as e:
        # We explicitly log startup failures using the infrastructure logger
        # Note: if _configure_logging hasn't run, this uses default logging, which is fine
        logging.getLogger("arbiter.bootstrap").critical(f"Startup failed: {e}")
        raise

    yield

    # Graceful Shutdown
    logger.info("Initiating graceful shutdown...")
    app.state.pipeline = None
    logger.info("Arbiter Pipeline reference released.")


def create_app() -> FastAPI:
    """Creates and configures the FastAPI application."""
    app = FastAPI(
        title="Arbiter API",
        description="HTTP API layer for the Arbiter pipeline",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.include_router(health.router)
    app.include_router(evaluation.router)

    @app.exception_handler(ArbiterError)
    async def arbiter_error_handler(
        request: Request, exc: ArbiterError
    ) -> JSONResponse:
        """Global exception handler mapping domain exceptions to HTTP responses."""
        # By default, ArbiterErrors are configuration or client issues for the API (e.g. ProfileNotFound)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": str(exc)},
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Global exception handler masking internal errors from external consumers."""
        logger.error(
            f"Unhandled exception during request processing: {exc}", exc_info=True
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal Server Error"},
        )

    return app


app = create_app()
