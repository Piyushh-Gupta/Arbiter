from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from src.core.pipeline.operations.operation_models import (
    PipelineHealthStatus,
    PipelineLifecycleState,
    PipelineOperationalMetadata,
    PipelineOperationalSnapshot,
    PipelineReadinessStatus,
    SubsystemHealthRecord,
)


@runtime_checkable
class BasePipelineHealthChecker(Protocol):
    """Protocol for aggregating pipeline health from subsystem records."""

    def evaluate_health(
        self, records: Sequence[SubsystemHealthRecord]
    ) -> PipelineHealthStatus:
        """Evaluates overall health based on individual subsystem health records."""
        ...


@runtime_checkable
class BasePipelineReadinessEvaluator(Protocol):
    """Protocol for evaluating pipeline readiness."""

    def evaluate_readiness(
        self,
        lifecycle_state: PipelineLifecycleState,
        health_status: PipelineHealthStatus,
        records: Sequence[SubsystemHealthRecord],
    ) -> PipelineReadinessStatus:
        """Evaluates if the pipeline is ready to accept traffic."""
        ...


@runtime_checkable
class BasePipelineLifecycleManager(Protocol):
    """Protocol for managing deterministic pipeline state transitions."""

    def transition_to(self, target_state: PipelineLifecycleState) -> None:
        """Attempts to transition the pipeline to a new state."""
        ...

    @property
    def current_state(self) -> PipelineLifecycleState:
        """Gets the current lifecycle state."""
        ...


@runtime_checkable
class BaseOperationalSnapshotBuilder(Protocol):
    """Protocol for constructing immutable operational snapshots."""

    def build_snapshot(
        self,
        lifecycle_state: PipelineLifecycleState,
        overall_health: PipelineHealthStatus,
        overall_readiness: PipelineReadinessStatus,
        subsystem_records: Sequence[SubsystemHealthRecord],
        metadata: PipelineOperationalMetadata,
    ) -> PipelineOperationalSnapshot:
        """Constructs an immutable pipeline operational snapshot."""
        ...


@runtime_checkable
class BasePipelineOperationsController(Protocol):
    """High-level orchestrator for pipeline operations."""

    def startup(self) -> None:
        """Executes graceful startup and state transitions."""
        ...

    def shutdown(self) -> None:
        """Executes graceful shutdown operations."""
        ...

    def get_snapshot(self) -> PipelineOperationalSnapshot:
        """Aggregates state into a comprehensive immutable snapshot."""
        ...
