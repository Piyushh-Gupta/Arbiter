"""Unit tests for health service."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

from src.api.services.health_service import HealthService
from src.api.services.service_models import (
    ClientMetadata,
    RequestMetadata,
    ServiceContext,
)
from src.core.pipeline.operations.operation_models import (
    PipelineHealthStatus,
    PipelineLifecycleState,
    PipelineOperationalMetadata,
    PipelineOperationalSnapshot,
    PipelineReadinessStatus,
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
        metadata=PipelineOperationalMetadata(environment="test", version="1.0.0"),
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
