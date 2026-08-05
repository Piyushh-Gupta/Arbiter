from collections.abc import Sequence

from src.core.pipeline.operations.base import BasePipelineHealthChecker
from src.core.pipeline.operations.operation_models import (
    PipelineHealthStatus,
    SubsystemHealthRecord,
)


class PipelineHealthChecker(BasePipelineHealthChecker):
    """Stateless evaluator of aggregate pipeline health."""

    def evaluate_health(
        self, records: Sequence[SubsystemHealthRecord]
    ) -> PipelineHealthStatus:
        """
        Evaluates overall health deterministically based on subsystem health records.
        - If any subsystem is UNHEALTHY, the pipeline is UNHEALTHY.
        - If any subsystem is DEGRADED (and none UNHEALTHY), the pipeline is DEGRADED.
        - Otherwise, HEALTHY.
        """
        if not records:
            return PipelineHealthStatus.HEALTHY

        has_degraded = False
        for record in records:
            if record.health_status == PipelineHealthStatus.UNHEALTHY:
                return PipelineHealthStatus.UNHEALTHY
            if record.health_status == PipelineHealthStatus.DEGRADED:
                has_degraded = True

        if has_degraded:
            return PipelineHealthStatus.DEGRADED

        return PipelineHealthStatus.HEALTHY
