from collections.abc import Sequence

from src.core.pipeline.operations.base import BasePipelineReadinessEvaluator
from src.core.pipeline.operations.operation_models import (
    PipelineHealthStatus,
    PipelineLifecycleState,
    PipelineReadinessStatus,
    SubsystemHealthRecord,
)


class PipelineReadinessEvaluator(BasePipelineReadinessEvaluator):
    """Stateless evaluator of pipeline readiness."""

    def __init__(self, require_all_subsystems_ready: bool = True) -> None:
        self._require_all_subsystems_ready = require_all_subsystems_ready

    def evaluate_readiness(
        self,
        lifecycle_state: PipelineLifecycleState,
        health_status: PipelineHealthStatus,
        records: Sequence[SubsystemHealthRecord],
    ) -> PipelineReadinessStatus:
        """
        Evaluates readiness deterministically based on lifecycle, health, and subsystems.
        The pipeline is only READY if:
        1. It is in the RUNNING state.
        2. It is not UNHEALTHY.
        3. All subsystems are READY (if configured to require it).
        """
        if lifecycle_state != PipelineLifecycleState.RUNNING:
            return PipelineReadinessStatus.NOT_READY

        if health_status == PipelineHealthStatus.UNHEALTHY:
            return PipelineReadinessStatus.NOT_READY

        if self._require_all_subsystems_ready:
            for record in records:
                if record.readiness_status == PipelineReadinessStatus.NOT_READY:
                    return PipelineReadinessStatus.NOT_READY

        return PipelineReadinessStatus.READY
