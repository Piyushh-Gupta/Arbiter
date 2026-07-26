"""Unit tests for the M15.2 FastAPI API Layer."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.core.exceptions import ArbiterError
from src.core.pipeline.pipeline_models import PipelineExecutionRequest


class MockEvaluationMetric:
    def __init__(self, identifier: str, title: str, score: float) -> None:
        self.identifier = identifier
        self.title = title
        self.score = score


class MockEvaluationResult:
    def __init__(self) -> None:
        self.metrics = [
            MockEvaluationMetric(identifier="mock", title="Mock", score=1.0)
        ]


class MockPipeline:
    def __init__(self, should_fail: bool = False) -> None:
        self.should_fail = should_fail
        self.last_request: PipelineExecutionRequest | None = None

    def execute(self, request: PipelineExecutionRequest) -> MockEvaluationResult:
        self.last_request = request
        if self.should_fail:
            raise ArbiterError("Simulated domain error")
        return MockEvaluationResult()


@pytest.fixture
def client() -> TestClient:
    app.state.pipeline = MockPipeline()
    return TestClient(app)


def test_health_check_live(client: TestClient) -> None:
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_health_check_ready(client: TestClient) -> None:
    response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_successful_evaluation(client: TestClient) -> None:
    payload = {
        "claim": "Test claim",
        "retrieval_profile_id": "1",
        "verification_profile_id": "2",
        "failure_analysis_profile_id": "3",
        "uncertainty_profile_id": "4",
        "decision_profile_id": "5",
        "explanation_profile_id": "6",
        "evaluation_profile_id": "7",
    }
    response = client.post("/v1/evaluate", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    assert len(data["metrics"]) == 1
    assert data["metrics"][0]["identifier"] == "mock"
    assert data["metrics"][0]["title"] == "Mock"
    assert data["metrics"][0]["score"] == 1.0


def test_validation_failure(client: TestClient) -> None:
    payload = {
        # missing claim
        "retrieval_profile_id": "1",
        "verification_profile_id": "2",
        "failure_analysis_profile_id": "3",
        "uncertainty_profile_id": "4",
        "decision_profile_id": "5",
        "explanation_profile_id": "6",
        "evaluation_profile_id": "7",
    }
    response = client.post("/v1/evaluate", json=payload)
    assert response.status_code == 422


def test_domain_exception_translation(client: TestClient) -> None:
    app.state.pipeline = MockPipeline(should_fail=True)
    payload = {
        "claim": "Test claim",
        "retrieval_profile_id": "1",
        "verification_profile_id": "2",
        "failure_analysis_profile_id": "3",
        "uncertainty_profile_id": "4",
        "decision_profile_id": "5",
        "explanation_profile_id": "6",
        "evaluation_profile_id": "7",
    }
    response = client.post("/v1/evaluate", json=payload)

    # Global exception handler maps ArbiterError to 400
    assert response.status_code == 400
    assert response.json() == {"detail": "Simulated domain error"}
