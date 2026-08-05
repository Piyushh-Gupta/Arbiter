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
        import datetime
        from unittest.mock import MagicMock

        from src.core.pipeline.operations.operation_models import (
            PipelineHealthStatus,
            PipelineLifecycleState,
            PipelineOperationalMetadata,
            PipelineOperationalSnapshot,
            PipelineReadinessStatus,
        )

        self.operations = MagicMock()
        self.operations.get_snapshot.return_value = PipelineOperationalSnapshot(
            timestamp=datetime.datetime.now(datetime.timezone.utc),
            lifecycle_state=PipelineLifecycleState.RUNNING,
            overall_health=PipelineHealthStatus.HEALTHY,
            overall_readiness=PipelineReadinessStatus.READY,
            subsystem_records=(),
            metadata=PipelineOperationalMetadata(environment="test", version="1.0.0"),
        )
        self.last_request: PipelineExecutionRequest | None = None

    def execute(self, request: PipelineExecutionRequest) -> MockEvaluationResult:
        self.last_request = request
        if self.should_fail:
            raise ArbiterError("Simulated domain error")
        return MockEvaluationResult()


@pytest.fixture
def client() -> TestClient:
    pipeline = MockPipeline()
    from src.api.services.factory import ServiceFactory

    app.state.pipeline = pipeline
    app.state.service_registry = ServiceFactory.build_registry(pipeline)  # type: ignore
    return TestClient(app)


def test_health_check_live(client: TestClient) -> None:
    response = client.get("/health/live")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "alive"
    assert "correlation_id" in data


def test_health_check_ready(client: TestClient) -> None:
    response = client.get("/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert "correlation_id" in data


def test_successful_evaluation(client: TestClient) -> None:
    payload = {
        "claim": "Test claim",
        "pipeline_profile_id": "1",
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
        "pipeline_profile_id": "1",
    }
    response = client.post("/v1/evaluate", json=payload)
    assert response.status_code == 422


def test_domain_exception_translation(client: TestClient) -> None:
    pipeline = MockPipeline(should_fail=True)
    from src.api.services.factory import ServiceFactory

    app.state.pipeline = pipeline
    app.state.service_registry = ServiceFactory.build_registry(pipeline)  # type: ignore
    payload = {
        "claim": "Test claim",
        "pipeline_profile_id": "1",
    }
    response = client.post("/v1/evaluate", json=payload)

    # Global exception handler maps ArbiterError to 400
    assert response.status_code == 400
    assert response.json() == {"detail": "Simulated domain error"}
