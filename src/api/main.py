from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.api.routes import evaluation, health
from src.core.bootstrap import build_pipeline
from src.core.config import Settings
from src.core.exceptions import ArbiterError


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lifespan events for the FastAPI application."""
    config = Settings()
    pipeline = build_pipeline(config)
    app.state.pipeline = pipeline
    yield


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
            status_code=400,
            content={"detail": str(exc)},
        )

    return app


app = create_app()
