import pytest

from src.core.exceptions import (
    IllegalLifecycleTransitionError,
    PipelineOperationalExecutionError,
)
from src.core.pipeline.operations.controller import PipelineOperationsController
from src.core.pipeline.operations.health import PipelineHealthChecker
from src.core.pipeline.operations.lifecycle import PipelineLifecycleManager
from src.core.pipeline.operations.operation_models import (
    PipelineHealthStatus,
    PipelineLifecycleState,
    PipelineOperationalMetadata,
    PipelineReadinessStatus,
    SubsystemHealthRecord,
)
from src.core.pipeline.operations.readiness import PipelineReadinessEvaluator
from src.core.pipeline.operations.snapshot import OperationalSnapshotBuilder


def test_lifecycle_manager_transitions() -> None:
    manager = PipelineLifecycleManager()
    assert manager.current_state == PipelineLifecycleState.STOPPED

    manager.transition_to(PipelineLifecycleState.INITIALIZING)
    assert manager.current_state == PipelineLifecycleState.INITIALIZING  # type: ignore

    manager.transition_to(PipelineLifecycleState.RUNNING)
    assert manager.current_state == PipelineLifecycleState.RUNNING

    manager.transition_to(PipelineLifecycleState.SHUTTING_DOWN)
    assert manager.current_state == PipelineLifecycleState.SHUTTING_DOWN

    manager.transition_to(PipelineLifecycleState.STOPPED)
    assert manager.current_state == PipelineLifecycleState.STOPPED


def test_lifecycle_manager_illegal_transition() -> None:
    manager = PipelineLifecycleManager()
    with pytest.raises(IllegalLifecycleTransitionError):
        manager.transition_to(PipelineLifecycleState.RUNNING)


def test_health_checker() -> None:
    checker = PipelineHealthChecker()
    # Empty records
    assert checker.evaluate_health([]) == PipelineHealthStatus.HEALTHY

    # Healthy records
    records = [
        SubsystemHealthRecord(
            subsystem_id="s1",
            health_status=PipelineHealthStatus.HEALTHY,
            readiness_status=PipelineReadinessStatus.READY,
        )
    ]
    assert checker.evaluate_health(records) == PipelineHealthStatus.HEALTHY

    # Degraded record
    records.append(
        SubsystemHealthRecord(
            subsystem_id="s2",
            health_status=PipelineHealthStatus.DEGRADED,
            readiness_status=PipelineReadinessStatus.READY,
        )
    )
    assert checker.evaluate_health(records) == PipelineHealthStatus.DEGRADED

    # Unhealthy record overrides degraded
    records.append(
        SubsystemHealthRecord(
            subsystem_id="s3",
            health_status=PipelineHealthStatus.UNHEALTHY,
            readiness_status=PipelineReadinessStatus.READY,
        )
    )
    assert checker.evaluate_health(records) == PipelineHealthStatus.UNHEALTHY


def test_readiness_evaluator() -> None:
    evaluator = PipelineReadinessEvaluator(require_all_subsystems_ready=True)

    records = [
        SubsystemHealthRecord(
            subsystem_id="s1",
            health_status=PipelineHealthStatus.HEALTHY,
            readiness_status=PipelineReadinessStatus.READY,
        )
    ]

    # Not running -> NOT_READY
    assert (
        evaluator.evaluate_readiness(
            PipelineLifecycleState.INITIALIZING, PipelineHealthStatus.HEALTHY, records
        )
        == PipelineReadinessStatus.NOT_READY
    )

    # Unhealthy -> NOT_READY
    assert (
        evaluator.evaluate_readiness(
            PipelineLifecycleState.RUNNING, PipelineHealthStatus.UNHEALTHY, records
        )
        == PipelineReadinessStatus.NOT_READY
    )

    # Running, healthy, all subsystems ready -> READY
    assert (
        evaluator.evaluate_readiness(
            PipelineLifecycleState.RUNNING, PipelineHealthStatus.HEALTHY, records
        )
        == PipelineReadinessStatus.READY
    )

    # One subsystem not ready -> NOT_READY
    records.append(
        SubsystemHealthRecord(
            subsystem_id="s2",
            health_status=PipelineHealthStatus.HEALTHY,
            readiness_status=PipelineReadinessStatus.NOT_READY,
        )
    )
    assert (
        evaluator.evaluate_readiness(
            PipelineLifecycleState.RUNNING, PipelineHealthStatus.HEALTHY, records
        )
        == PipelineReadinessStatus.NOT_READY
    )

    # Test without requiring all subsystems ready
    evaluator_loose = PipelineReadinessEvaluator(require_all_subsystems_ready=False)
    assert (
        evaluator_loose.evaluate_readiness(
            PipelineLifecycleState.RUNNING, PipelineHealthStatus.HEALTHY, records
        )
        == PipelineReadinessStatus.READY
    )


def test_snapshot_builder() -> None:
    builder = OperationalSnapshotBuilder()
    metadata = PipelineOperationalMetadata(environment="test", version="1.0.0")
    records = [
        SubsystemHealthRecord(
            subsystem_id="s1",
            health_status=PipelineHealthStatus.HEALTHY,
            readiness_status=PipelineReadinessStatus.READY,
        )
    ]
    snapshot = builder.build_snapshot(
        lifecycle_state=PipelineLifecycleState.RUNNING,
        overall_health=PipelineHealthStatus.HEALTHY,
        overall_readiness=PipelineReadinessStatus.READY,
        subsystem_records=records,
        metadata=metadata,
    )
    assert snapshot.lifecycle_state == PipelineLifecycleState.RUNNING
    assert snapshot.overall_health == PipelineHealthStatus.HEALTHY
    assert snapshot.overall_readiness == PipelineReadinessStatus.READY
    assert len(snapshot.subsystem_records) == 1
    assert snapshot.metadata.environment == "test"


def test_pipeline_operations_controller_startup_shutdown() -> None:
    metadata = PipelineOperationalMetadata(environment="test", version="1.0.0")
    controller = PipelineOperationsController(
        lifecycle_manager=PipelineLifecycleManager(),
        health_checker=PipelineHealthChecker(),
        readiness_evaluator=PipelineReadinessEvaluator(),
        snapshot_builder=OperationalSnapshotBuilder(),
        metadata=metadata,
        subsystem_record_provider=lambda: [],
    )

    controller.startup()
    assert controller._lifecycle.current_state == PipelineLifecycleState.RUNNING

    snapshot = controller.get_snapshot()
    assert snapshot.lifecycle_state == PipelineLifecycleState.RUNNING
    assert snapshot.overall_health == PipelineHealthStatus.HEALTHY
    assert snapshot.overall_readiness == PipelineReadinessStatus.READY

    controller.shutdown()
    assert controller._lifecycle.current_state == PipelineLifecycleState.STOPPED  # type: ignore


def test_pipeline_operations_controller_error_handling() -> None:
    metadata = PipelineOperationalMetadata(environment="test", version="1.0.0")
    controller = PipelineOperationsController(
        lifecycle_manager=PipelineLifecycleManager(),
        health_checker=PipelineHealthChecker(),
        readiness_evaluator=PipelineReadinessEvaluator(),
        snapshot_builder=OperationalSnapshotBuilder(),
        metadata=metadata,
    )

    # Trying to shutdown while stopped
    with pytest.raises(PipelineOperationalExecutionError):
        controller.shutdown()

    controller.startup()

    # Trying to startup while running
    with pytest.raises(PipelineOperationalExecutionError):
        controller.startup()


def test_pipeline_operations_controller_provider_error() -> None:
    def failing_provider() -> list[SubsystemHealthRecord]:
        raise ValueError("Provider error")

    metadata = PipelineOperationalMetadata(environment="test", version="1.0.0")
    controller = PipelineOperationsController(
        lifecycle_manager=PipelineLifecycleManager(),
        health_checker=PipelineHealthChecker(),
        readiness_evaluator=PipelineReadinessEvaluator(),
        snapshot_builder=OperationalSnapshotBuilder(),
        metadata=metadata,
        subsystem_record_provider=failing_provider,
    )

    with pytest.raises(
        PipelineOperationalExecutionError, match="Failed to retrieve subsystem records"
    ):
        controller.startup()
