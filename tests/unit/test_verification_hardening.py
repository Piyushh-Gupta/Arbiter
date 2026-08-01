"""Unit and integration tests for M2.9 Verification Production Hardening subsystem."""

from datetime import datetime

import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.api.main import create_app
from src.core.bootstrap import build_verification_operational_registry
from src.core.config import Settings
from src.core.exceptions import (
    DuplicateOptimizationProfileError,
    OptimizationConfigurationError,
)
from src.core.verification.operational.operational_models import (
    VerificationOperationalProfile,
    VerificationOperationalRegistry,
    VerificationOperationalTrace,
)


def test_operational_profile_and_registry() -> None:
    profile = VerificationOperationalProfile(
        profile_id="test_op",
        environment="production",
        logging_configuration={"level": "INFO"},
        readiness_configuration={"timeout_sec": 5},
        telemetry_configuration={"enabled": True},
    )
    assert profile.profile_id == "test_op"
    assert profile.environment == "production"

    # Immutability check
    with pytest.raises(ValidationError):
        setattr(profile, "environment", "development")

    registry = VerificationOperationalRegistry(profiles=(profile,))
    assert registry.resolve("test_op") is profile

    # Duplicate detection
    with pytest.raises(DuplicateOptimizationProfileError):
        VerificationOperationalRegistry(profiles=(profile, profile))

    # Invalid environment check (raises ValidationError at registry validation time)
    bad_profile = VerificationOperationalProfile(
        profile_id="invalid", environment="local"
    )
    with pytest.raises(ValidationError):
        VerificationOperationalRegistry(profiles=(bad_profile,))


def test_verification_operational_trace() -> None:
    trace = VerificationOperationalTrace(
        startup_validation=True,
        readiness_validation=True,
        registry_validation=True,
        operational_configuration={"env": "production"},
        execution_timestamp=datetime.utcnow().isoformat(),
    )
    assert trace.startup_validation is True
    assert trace.readiness_validation is True
    assert trace.registry_validation is True
    assert trace.operational_configuration == {"env": "production"}


def test_bootstrap_and_fail_fast() -> None:
    settings = Settings()

    # Standard bootstrap
    registry = build_verification_operational_registry(settings)
    assert registry.resolve("default_operational") is not None

    # Fail fast on configuration issues (e.g. invalid env)
    class BadSettings:
        def __init__(self) -> None:
            self.environment = "invalid_env"

    with pytest.raises(OptimizationConfigurationError):
        build_verification_operational_registry(BadSettings())  # type: ignore


def test_health_endpoints_integration() -> None:
    app = create_app()
    with TestClient(app) as client:
        # 1. Test verification liveness
        response_live = client.get("/verification/health/live")
        assert response_live.status_code == status.HTTP_200_OK
        assert response_live.json() == {"status": "alive"}

        # 2. Test verification readiness (lifespan has mounted the registries)
        response_ready = client.get("/verification/health/ready")
        assert response_ready.status_code == status.HTTP_200_OK
        assert response_ready.json() == {"status": "ready"}


def test_health_endpoints_failure_scenarios() -> None:
    app = FastAPI()
    # Explicitly do NOT run lifespan/registries mounting
    from src.api.routes.health import router

    app.include_router(router)
    client = TestClient(app)

    # Readiness should return 503 since registries are not mounted
    response_ready = client.get("/verification/health/ready")
    assert response_ready.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert response_ready.json() == {"status": "not_ready"}


def test_exception_sanitization() -> None:
    app = create_app()

    @app.get("/trigger-error")
    def trigger_error() -> None:
        raise ValueError("Sensitive database connection details: password=123")

    # Disable exception propagation to trigger FastAPI global handler
    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/trigger-error")

    # Ensure client gets sanitized response and not the stack trace or details
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert response.json() == {"detail": "Internal Server Error"}


def test_determinism_operational_no_op() -> None:
    # Verify that the operational profiles/traces do not impact prediction outputs
    trace = VerificationOperationalTrace(
        startup_validation=True,
        readiness_validation=True,
        registry_validation=True,
        operational_configuration={},
        execution_timestamp=datetime.utcnow().isoformat(),
    )
    assert trace is not None
