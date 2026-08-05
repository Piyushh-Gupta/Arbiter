import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.api.routes import evaluation, health
from src.api.services.exceptions import ExceptionTranslator
from src.core.bootstrap import (
    build_calibration_registry,
    build_contract_engine,
    build_contract_registry,
    build_pipeline,
    build_pipeline_benchmark_registry,
    build_pipeline_explanation_registry,
    build_resilience_controller,
    build_resilience_registry,
    build_services,
    build_telemetry_engine,
    build_verification_operational_registry,
    build_verification_optimization_registry,
    initialize_application,
)
from src.core.config import Settings
from src.core.exceptions import ArbiterError

logger = logging.getLogger("arbiter.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lifespan events for the FastAPI application."""
    config = Settings()
    try:
        initialize_application(config)
        telemetry_engine = build_telemetry_engine(config)

        # Set up pipeline resilience dependencies
        resilience_executor = ThreadPoolExecutor(max_workers=4)
        resilience_registry = build_resilience_registry(config, resilience_executor)
        resilience_controller = build_resilience_controller(config, resilience_executor)
        resilience_profile = resilience_registry.resolve(
            config.pipeline_resilience.active_resilience_profile_id
        )

        pipeline = build_pipeline(
            config,
            telemetry_hook=telemetry_engine.observe,
            resilience_controller=resilience_controller,
            resilience_profile=resilience_profile,
        )
        app.state.pipeline = pipeline
        if hasattr(pipeline, "operations") and pipeline.operations is not None:
            pipeline.operations.startup()

        app.state.service_registry = build_services(config, pipeline)

        contract_registry = build_contract_registry(config)
        app.state.contract_engine = build_contract_engine(config, contract_registry)

        app.state.telemetry_engine = telemetry_engine
        app.state.resilience_executor = resilience_executor
        app.state.resilience_registry = resilience_registry
        app.state.resilience_controller = resilience_controller
        app.state.verification_optimization_registry = (
            build_verification_optimization_registry(config)
        )
        app.state.verification_operational_registry = (
            build_verification_operational_registry(config)
        )
        app.state.calibration_registry = build_calibration_registry(config)
        app.state.pipeline_benchmark_registry = build_pipeline_benchmark_registry(
            config
        )
        app.state.pipeline_explanation_registry = build_pipeline_explanation_registry(
            config
        )
        logger.info(
            "Arbiter Pipeline, telemetry engine, resilience engine, benchmark registry, explanation registry, and optimization/operational/calibration registries mounted successfully."
        )
    except Exception as e:
        # We explicitly log startup failures using the infrastructure logger
        logging.getLogger("arbiter.bootstrap").critical(f"Startup failed: {e}")
        raise

    yield

    # Graceful Shutdown
    logger.info("Initiating graceful shutdown...")
    if hasattr(app.state, "pipeline") and app.state.pipeline is not None:
        if (
            hasattr(app.state.pipeline, "_operations_controller")
            and app.state.pipeline._operations_controller is not None
        ):
            try:
                app.state.pipeline._operations_controller.shutdown()
            except Exception as e:
                logger.error(f"Error during pipeline operational shutdown: {e}")
    if (
        hasattr(app.state, "resilience_executor")
        and app.state.resilience_executor is not None
    ):
        app.state.resilience_executor.shutdown(wait=True)
    app.state.contract_engine = None
    app.state.service_registry = None
    app.state.pipeline = None
    app.state.telemetry_engine = None
    app.state.resilience_executor = None
    app.state.resilience_registry = None
    app.state.resilience_controller = None
    app.state.verification_optimization_registry = None
    app.state.verification_operational_registry = None
    app.state.calibration_registry = None
    app.state.pipeline_benchmark_registry = None
    app.state.pipeline_explanation_registry = None
    logger.info(
        "Arbiter Pipeline reference, telemetry engine, resilience executor/registry, benchmark registry, explanation registry, and operational/calibration references released."
    )


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
        translated = ExceptionTranslator.translate(exc)
        return JSONResponse(
            status_code=translated.status_code,
            content={"detail": translated.detail},
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Global exception handler masking internal errors from external consumers."""
        logger.error(
            f"Unhandled exception during request processing: {exc}", exc_info=True
        )
        translated = ExceptionTranslator.translate(exc)
        return JSONResponse(
            status_code=translated.status_code,
            content={"detail": translated.detail},
        )

    return app


app = create_app()
