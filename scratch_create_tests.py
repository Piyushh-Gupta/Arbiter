import os

os.makedirs("tests/unit/api/services", exist_ok=True)

with open("tests/unit/api/services/test_service_models.py", "w", encoding="utf-8") as f:
    f.write('''"""Unit tests for service models."""

import pytest
from pydantic import ValidationError

from src.api.services.service_models import (
    ClientMetadata,
    EvaluationRequest,
    EvaluationResponse,
    HealthStatusResponse,
    MetricServiceDTO,
    RequestMetadata,
    ServiceContext,
    ServiceExecutionMetadata,
)


def test_request_metadata_frozen() -> None:
    rm = RequestMetadata(headers={"x-foo": "bar"}, query_params={"q": "1"}, client_ip="127.0.0.1")
    with pytest.raises(ValidationError):
        rm.client_ip = "192.168.1.1"


def test_service_context_initialization() -> None:
    ctx = ServiceContext(
        correlation_id="1234",
        request_metadata=RequestMetadata(),
        client_metadata=ClientMetadata(),
    )
    assert ctx.correlation_id == "1234"
    assert ctx.request_metadata.client_ip is None


def test_evaluation_request_validation() -> None:
    with pytest.raises(ValidationError):
        EvaluationRequest(claim="test", pipeline_profile_id="default")  # missing context
''')

with open("tests/unit/api/services/test_evaluation_service.py", "w", encoding="utf-8") as f:
    f.write('''"""Unit tests for evaluation service."""

from unittest.mock import MagicMock

from src.api.services.evaluation_service import EvaluationService
from src.api.services.service_models import (
    ClientMetadata,
    EvaluationRequest,
    RequestMetadata,
    ServiceContext,
)
from src.core.pipeline.orchestrator import ArbiterPipeline
from src.core.pipeline.pipeline_models import PipelineExecutionResult
from src.core.evaluation.evaluation_models import PipelineEvaluationMetric


def test_evaluation_service_success() -> None:
    mock_pipeline = MagicMock(spec=ArbiterPipeline)
    mock_metric = PipelineEvaluationMetric(identifier="test", title="Test", score=1.0)
    mock_pipeline.execute.return_value = PipelineExecutionResult(
        claim="test",
        pipeline_profile_id="default",
        metrics=(mock_metric,)
    )

    service = EvaluationService(pipeline=mock_pipeline)
    
    req = EvaluationRequest(
        claim="test",
        pipeline_profile_id="default",
        context=ServiceContext(
            correlation_id="req-123",
            request_metadata=RequestMetadata(),
            client_metadata=ClientMetadata(),
        )
    )
    
    resp = service.evaluate(req)
    
    assert resp.correlation_id == "req-123"
    assert len(resp.metrics) == 1
    assert resp.metrics[0].identifier == "test"
    assert resp.execution_metadata.duration_ms > 0
''')

with open("tests/unit/api/services/test_health_service.py", "w", encoding="utf-8") as f:
    f.write('''"""Unit tests for health service."""

from unittest.mock import MagicMock

from src.api.services.health_service import HealthService
from src.api.services.service_models import (
    ClientMetadata,
    RequestMetadata,
    ServiceContext,
)
from src.core.pipeline.operations.operation_models import (
    PipelineOperationalSnapshot,
    PipelineReadinessStatus,
    PipelineLifecycleState,
    PipelineHealthStatus,
)
from src.core.pipeline.orchestrator import ArbiterPipeline


def test_health_service_liveness() -> None:
    mock_pipeline = MagicMock(spec=ArbiterPipeline)
    service = HealthService(pipeline=mock_pipeline)
    
    ctx = ServiceContext(
        correlation_id="ctx-1",
        request_metadata=RequestMetadata(),
        client_metadata=ClientMetadata(),
    )
    
    resp = service.check_liveness(ctx)
    assert resp.status == "alive"
    assert resp.correlation_id == "ctx-1"


def test_health_service_readiness_ready() -> None:
    mock_pipeline = MagicMock(spec=ArbiterPipeline)
    mock_operations = MagicMock()
    mock_snapshot = PipelineOperationalSnapshot(
        snapshot_id="1",
        timestamp_ns=1,
        lifecycle_state=PipelineLifecycleState.RUNNING,
        health_status=PipelineHealthStatus.HEALTHY,
        readiness_status=PipelineReadinessStatus.READY,
        subsystems=(),
    )
    mock_operations.get_snapshot.return_value = mock_snapshot
    mock_pipeline.operations = mock_operations
    
    service = HealthService(pipeline=mock_pipeline)
    
    ctx = ServiceContext(
        correlation_id="ctx-2",
        request_metadata=RequestMetadata(),
        client_metadata=ClientMetadata(),
    )
    
    resp = service.check_readiness(ctx)
    assert resp.status == "ready"
    assert resp.correlation_id == "ctx-2"


def test_health_service_readiness_not_ready() -> None:
    mock_pipeline = MagicMock(spec=ArbiterPipeline)
    # mock_pipeline does not have operations
    del mock_pipeline.operations
    
    service = HealthService(pipeline=mock_pipeline)
    
    ctx = ServiceContext(
        correlation_id="ctx-3",
        request_metadata=RequestMetadata(),
        client_metadata=ClientMetadata(),
    )
    
    resp = service.check_readiness(ctx)
    assert resp.status == "not_ready"
    assert resp.correlation_id == "ctx-3"
''')
