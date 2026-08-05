"""Unit tests for evaluation service."""

from unittest.mock import MagicMock

from src.api.services.evaluation_service import EvaluationService
from src.api.services.service_models import (
    ClientMetadata,
    EvaluationRequest,
    RequestMetadata,
    ServiceContext,
)
from src.core.evaluation.evaluation_models import EvaluationMetric
from src.core.pipeline.orchestrator import ArbiterPipeline


def test_evaluation_service_success() -> None:
    mock_pipeline = MagicMock(spec=ArbiterPipeline)
    mock_metric = EvaluationMetric(identifier="test", title="Test", score=1.0)
    mock_pipeline.execute.return_value = MagicMock(metrics=(mock_metric,))

    service = EvaluationService(pipeline=mock_pipeline)

    req = EvaluationRequest(
        claim="test",
        pipeline_profile_id="default",
        context=ServiceContext(
            correlation_id="req-123",
            request_metadata=RequestMetadata(),
            client_metadata=ClientMetadata(),
        ),
    )

    resp = service.evaluate(req)

    assert resp.correlation_id == "req-123"
    assert len(resp.metrics) == 1
    assert resp.metrics[0].identifier == "test"
    assert resp.execution_metadata.duration_ms > 0
