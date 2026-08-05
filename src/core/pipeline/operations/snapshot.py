from collections.abc import Sequence

from src.core.pipeline.operations.base import BaseOperationalSnapshotBuilder
from src.core.pipeline.operations.operation_models import (
    PipelineHealthStatus,
    PipelineLifecycleState,
    PipelineOperationalMetadata,
    PipelineOperationalSnapshot,
    PipelineReadinessStatus,
    SubsystemHealthRecord,
)


class OperationalSnapshotBuilder(BaseOperationalSnapshotBuilder):
    """Stateless builder for constructing immutable operational snapshots."""

    def build_snapshot(
        self,
        lifecycle_state: PipelineLifecycleState,
        overall_health: PipelineHealthStatus,
        overall_readiness: PipelineReadinessStatus,
        subsystem_records: Sequence[SubsystemHealthRecord],
        metadata: PipelineOperationalMetadata,
    ) -> PipelineOperationalSnapshot:
        """Constructs an immutable pipeline operational snapshot deterministically."""
        return PipelineOperationalSnapshot(
            lifecycle_state=lifecycle_state,
            overall_health=overall_health,
            overall_readiness=overall_readiness,
            subsystem_records=tuple(subsystem_records),
            metadata=metadata,
        )
