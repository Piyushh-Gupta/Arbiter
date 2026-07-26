"""HTTP integration tests for the Arbiter Pipeline."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.core.config import Settings


def test_health_endpoint(app: FastAPI) -> None:
    """Verifies the health check endpoint returns 200 OK."""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def test_successful_pipeline_execution(
    app: FastAPI, valid_http_request_payload: dict[str, str]
) -> None:
    """
    Verifies that a valid POST /v1/evaluate request properly triggers the
    FastAPI router -> pipeline -> subsystem engines -> HTTP Response.
    """
    with TestClient(app) as client:
        # FastAPI lifespan mounts app.state.pipeline
        assert hasattr(app.state, "pipeline")

        response = client.post("/v1/evaluate", json=valid_http_request_payload)

        assert response.status_code == 200
        data = response.json()

        # Verify DTO serialization
        assert "metrics" in data
        assert isinstance(data["metrics"], list)
        assert len(data["metrics"]) > 0

        # Ensure determinism across identical HTTP requests
        response_two = client.post("/v1/evaluate", json=valid_http_request_payload)
        assert response.json() == response_two.json()


def test_invalid_request_validation(app: FastAPI) -> None:
    """Verifies that malformed HTTP requests yield 422 Unprocessable Entity."""
    with TestClient(app) as client:
        # Missing required fields
        invalid_payload = {"claim": "I have no profiles"}
        response = client.post("/v1/evaluate", json=invalid_payload)
        assert response.status_code == 422

        # Empty claim
        empty_claim_payload = {
            "claim": "",
            "retrieval_profile_id": "default_retrieval",
            "verification_profile_id": "default_verification",
            "failure_analysis_profile_id": "default_failure_analysis",
            "uncertainty_profile_id": "default_uncertainty",
            "decision_profile_id": "default_decision",
            "explanation_profile_id": "default_explanation",
            "evaluation_profile_id": "default_evaluation",
        }
        response_empty = client.post("/v1/evaluate", json=empty_claim_payload)
        assert response_empty.status_code == 422


def test_exception_translation(
    app: FastAPI, valid_http_request_payload: dict[str, str]
) -> None:
    """
    Verifies that domain exceptions (like ProfileNotFoundError) correctly
    bubble up and translate into sanitized HTTP 400 responses.
    """
    with TestClient(app) as client:
        # Mutate the payload to request an unknown profile
        payload = dict(valid_http_request_payload)
        payload["retrieval_profile_id"] = "non_existent_profile"

        response = client.post("/v1/evaluate", json=payload)
        assert response.status_code == 400

        data = response.json()
        assert "detail" in data
        assert "non_existent_profile" in data["detail"]


def test_startup_configuration_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that an invalid configuration violently crashes the application
    during the lifespan boot sequence, ensuring a fail-fast startup.
    """
    # Force an invalid type to intentionally corrupt Pydantic Settings
    # max_retries expects an int, we pass a string
    monkeypatch.setenv("download__max_retries", "INVALID")

    with pytest.raises(ValidationError):
        # Attempting to load settings should immediately trigger a validation error
        Settings()


def test_startup_incompatible_profile(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Verifies that the application fails to start if an incompatible profile/engine
    pairing is registered in the composition root.
    """
    from src.core.evaluation.base import BaseEvaluator
    from src.core.evaluation.evaluation_models import (
        EvaluationDefinition,
        EvaluationProfile,
        EvaluationResult,
        RuleBasedEvaluationDefinition,
    )
    from src.core.exceptions import EvaluationConfigurationError
    from src.core.explainability.explainability_models import ExplanationResult

    class IncompatibleEvaluator(BaseEvaluator):
        def validate_compatibility(self, definition: EvaluationDefinition) -> None:
            raise EvaluationConfigurationError("Incompatible definition")

        def evaluate(
            self,
            explanation_result: ExplanationResult,
            definition: EvaluationDefinition,
        ) -> EvaluationResult:
            raise NotImplementedError()

    with pytest.raises((ValidationError, EvaluationConfigurationError)) as exc_info:
        # Pydantic model_validator will catch this at construction
        EvaluationProfile(
            profile_id="bad_profile",
            definition=RuleBasedEvaluationDefinition(),
            engine=IncompatibleEvaluator(),
        )
    assert "Incompatible definition" in str(exc_info.value)
