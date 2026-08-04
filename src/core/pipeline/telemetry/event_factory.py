"""Default telemetry event factory implementation."""

from src.core.pipeline.pipeline_models import PipelineExecutionResult
from src.core.pipeline.telemetry.base import BaseTelemetryEventFactory
from src.core.pipeline.telemetry.telemetry_models import (
    PipelineStageTelemetryRecord,
    PipelineTelemetryEvent,
)


class DefaultTelemetryEventFactory(BaseTelemetryEventFactory):
    """Factory to build PipelineTelemetryEvent from PipelineExecutionResult."""

    def from_result(self, result: PipelineExecutionResult) -> PipelineTelemetryEvent:
        """Constructs a PipelineTelemetryEvent from an immutable PipelineExecutionResult."""
        ctx = result.execution_context
        stage_records = tuple(
            PipelineStageTelemetryRecord(
                stage_id=stage.stage_id,
                profile_id=stage.profile_id,
                latency_ms=stage.latency_ms,
                success=stage.success,
            )
            for stage in ctx.stage_metadata
        )

        return PipelineTelemetryEvent(
            execution_id=ctx.execution_id,
            pipeline_id=ctx.pipeline_id,
            claim_length=len(ctx.claim),
            total_latency_ms=ctx.total_latency_ms,
            success=ctx.success,
            stage_records=stage_records,
            configuration_fingerprint=ctx.runtime_metadata.configuration_fingerprint,
            execution_environment=ctx.runtime_metadata.execution_environment,
            observed_at=ctx.runtime_metadata.execution_timestamp,
            schema_version=ctx.runtime_metadata.schema_version,
        )
