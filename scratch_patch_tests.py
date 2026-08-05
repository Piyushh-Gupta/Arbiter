with open("src/api/services/evaluation_service.py", "r", encoding="utf-8") as f:
    code = f.read()

code = code.replace("for m in domain_res.metrics", "for m in domain_res.evaluation_result.metrics")

with open("src/api/services/evaluation_service.py", "w", encoding="utf-8") as f:
    f.write(code)


with open("tests/unit/api/services/test_service_models.py", "r", encoding="utf-8") as f:
    code = f.read()

code = code.replace("EvaluationRequest(claim=\"test\", pipeline_profile_id=\"default\")  # missing context", "EvaluationRequest(claim=\"test\", pipeline_profile_id=\"default\")  # type: ignore")

with open("tests/unit/api/services/test_service_models.py", "w", encoding="utf-8") as f:
    f.write(code)

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
from src.core.evaluation.evaluation_models import EvaluationMetric


def test_evaluation_service_success() -> None:
    mock_pipeline = MagicMock(spec=ArbiterPipeline)
    mock_metric = EvaluationMetric(identifier="test", title="Test", score=1.0)
    mock_pipeline.execute.return_value = MagicMock(
        evaluation_result=MagicMock(metrics=(mock_metric,))
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

from datetime import datetime, timezone
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
        timestamp=datetime.now(timezone.utc),
        lifecycle_state=PipelineLifecycleState.RUNNING,
        overall_health=PipelineHealthStatus.HEALTHY,
        overall_readiness=PipelineReadinessStatus.READY,
        subsystem_records=(),
        metadata=MagicMock(),
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
