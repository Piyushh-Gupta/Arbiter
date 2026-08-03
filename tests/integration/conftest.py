"""Shared fixtures for integration tests."""

import pytest
from fastapi import FastAPI

from src.api.main import app as fastapi_app
from src.core.pipeline.pipeline_models import PipelineExecutionRequest


@pytest.fixture
def app() -> FastAPI:
    """Provides the FastAPI application."""
    return fastapi_app


@pytest.fixture
def valid_http_request_payload() -> dict[str, str]:
    """Provides a valid EvaluateClaimRequest payload as a dictionary."""
    return {
        "claim": "The quick brown fox jumps over the lazy dog.",
        "pipeline_profile_id": "default_pipeline",
    }


@pytest.fixture
def valid_domain_request() -> PipelineExecutionRequest:
    """Provides a valid PipelineExecutionRequest domain model."""
    return PipelineExecutionRequest(
        claim="The Eiffel Tower is located in Paris.",
        pipeline_profile_id="default_pipeline",
    )
