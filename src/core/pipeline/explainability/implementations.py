"""Concrete explanation strategies for Pipeline Explainability (M5.5)."""

from src.core.pipeline.explainability.base import BasePipelineExplanationStrategy
from src.core.pipeline.explainability.explainability_models import (
    PipelineDecisionTrace,
    PipelineExecutionExplanation,
    PipelineExecutionSummary,
    PipelineExplanationDefinition,
    PipelineExplanationInput,
    PipelineStageExplanation,
    PipelineTelemetryContext,
)
from src.core.pipeline.explainability.utils import generate_sha256_trace_id


class SummaryExplanationStrategy(BasePipelineExplanationStrategy):
    """Produces a high-level summary of a pipeline execution."""

    @property
    def strategy_id(self) -> str:
        return "pipeline_summary"

    def validate_compatibility(self, definition: PipelineExplanationDefinition) -> None:
        pass

    def generate_explanation(
        self,
        input_data: PipelineExplanationInput,
        definition: PipelineExplanationDefinition,
    ) -> PipelineExecutionExplanation:
        context = input_data.execution_result.execution_context
        runtime = context.runtime_metadata

        outcome = "SUCCESS" if context.success else "FAILURE"
        summary = PipelineExecutionSummary(
            outcome=outcome,
            total_latency_ms=context.total_latency_ms,
            stage_count=len(context.stage_metadata),
            configuration_fingerprint=runtime.configuration_fingerprint,
        )

        trace_id = generate_sha256_trace_id(
            f"{context.execution_id}_{context.pipeline_id}"
        )

        return PipelineExecutionExplanation(
            execution_id=context.execution_id,
            pipeline_id=context.pipeline_id,
            claim_length=len(context.claim),
            success=context.success,
            total_latency_ms=context.total_latency_ms,
            configuration_fingerprint=runtime.configuration_fingerprint,
            schema_version=runtime.schema_version,
            execution_environment=runtime.execution_environment,
            summary=summary,
            stage_explanations=(),
            decision_trace=None,
            telemetry_context=PipelineTelemetryContext(),
            metadata={"trace_id": trace_id},
        )


class ExecutionTraceStrategy(BasePipelineExplanationStrategy):
    """Produces a detailed resilience decision trace for pipeline execution."""

    @property
    def strategy_id(self) -> str:
        return "pipeline_trace"

    def validate_compatibility(self, definition: PipelineExplanationDefinition) -> None:
        pass

    def generate_explanation(
        self,
        input_data: PipelineExplanationInput,
        definition: PipelineExplanationDefinition,
    ) -> PipelineExecutionExplanation:
        context = input_data.execution_result.execution_context
        runtime = context.runtime_metadata

        decision_trace = None
        res_meta = getattr(input_data.execution_result, "resilience_metadata", None)
        if res_meta is not None:
            retry_trace = res_meta.retry_trace
            succeeded_on_attempt = (
                retry_trace.total_attempts if retry_trace.succeeded else None
            )
            trace_id = generate_sha256_trace_id(
                f"resilience_{retry_trace.execution_id}"
            )

            decision_trace = PipelineDecisionTrace(
                total_attempts=retry_trace.total_attempts,
                succeeded_on_attempt=succeeded_on_attempt,
                timeout_enforced=res_meta.timeout_enforced,
                recovery_invoked=res_meta.recovery_invoked,
                recovery_strategy_id=res_meta.recovery_strategy_id,
                terminal_error=retry_trace.terminal_error,
                total_retry_overhead_ms=retry_trace.total_retry_overhead_ms,
                trace_id=trace_id,
            )

        summary = PipelineExecutionSummary(
            outcome="SUCCESS" if context.success else "FAILURE",
            total_latency_ms=context.total_latency_ms,
            stage_count=len(context.stage_metadata),
            configuration_fingerprint=runtime.configuration_fingerprint,
        )

        trace_id = generate_sha256_trace_id(f"trace_{context.execution_id}")

        return PipelineExecutionExplanation(
            execution_id=context.execution_id,
            pipeline_id=context.pipeline_id,
            claim_length=len(context.claim),
            success=context.success,
            total_latency_ms=context.total_latency_ms,
            configuration_fingerprint=runtime.configuration_fingerprint,
            schema_version=runtime.schema_version,
            execution_environment=runtime.execution_environment,
            summary=summary,
            stage_explanations=(),
            decision_trace=decision_trace,
            telemetry_context=PipelineTelemetryContext(),
            metadata={"trace_id": trace_id},
        )


class StageBreakdownStrategy(BasePipelineExplanationStrategy):
    """Produces per-stage latency breakdown and telemetry percentile ranks."""

    @property
    def strategy_id(self) -> str:
        return "pipeline_stage_breakdown"

    def validate_compatibility(self, definition: PipelineExplanationDefinition) -> None:
        pass

    def generate_explanation(
        self,
        input_data: PipelineExplanationInput,
        definition: PipelineExplanationDefinition,
    ) -> PipelineExecutionExplanation:
        context = input_data.execution_result.execution_context
        runtime = context.runtime_metadata

        stage_exps = []
        for sm in context.stage_metadata:
            rank = None
            if input_data.telemetry_snapshot is not None:
                aggs = input_data.telemetry_snapshot.stage_aggregations
                if aggs:
                    lower_count = sum(
                        1 for agg in aggs if agg.mean_latency_ms < sm.latency_ms
                    )
                    rank = (lower_count / len(aggs)) * 100.0

            observation = (
                f"Stage '{sm.stage_id}' completed in {sm.latency_ms:.2f}ms "
                f"({'success' if sm.success else 'failure'})"
            )
            stage_trace_id = generate_sha256_trace_id(
                f"{sm.stage_id}_{sm.profile_id}_{sm.latency_ms:.4f}_{sm.success}"
            )

            stage_exps.append(
                PipelineStageExplanation(
                    stage_id=sm.stage_id,
                    profile_id=sm.profile_id,
                    latency_ms=sm.latency_ms,
                    success=sm.success,
                    latency_percentile_rank=rank,
                    observation=observation,
                    trace_id=stage_trace_id,
                )
            )

        telemetry_context = PipelineTelemetryContext()
        if input_data.telemetry_snapshot is not None:
            telemetry_context = PipelineTelemetryContext(
                total_executions=input_data.telemetry_snapshot.total_executions,
                overall_success_rate=input_data.telemetry_snapshot.overall_success_rate,
            )

        summary = PipelineExecutionSummary(
            outcome="SUCCESS" if context.success else "FAILURE",
            total_latency_ms=context.total_latency_ms,
            stage_count=len(context.stage_metadata),
            configuration_fingerprint=runtime.configuration_fingerprint,
        )

        trace_id = generate_sha256_trace_id(f"breakdown_{context.execution_id}")

        return PipelineExecutionExplanation(
            execution_id=context.execution_id,
            pipeline_id=context.pipeline_id,
            claim_length=len(context.claim),
            success=context.success,
            total_latency_ms=context.total_latency_ms,
            configuration_fingerprint=runtime.configuration_fingerprint,
            schema_version=runtime.schema_version,
            execution_environment=runtime.execution_environment,
            summary=summary,
            stage_explanations=tuple(stage_exps),
            decision_trace=None,
            telemetry_context=telemetry_context,
            metadata={"trace_id": trace_id},
        )


class CompositePipelineExplanationStrategy(BasePipelineExplanationStrategy):
    """Combines summary, trace, and stage breakdown strategies into a unified explanation."""

    @property
    def strategy_id(self) -> str:
        return "pipeline_composite"

    def validate_compatibility(self, definition: PipelineExplanationDefinition) -> None:
        pass

    def generate_explanation(
        self,
        input_data: PipelineExplanationInput,
        definition: PipelineExplanationDefinition,
    ) -> PipelineExecutionExplanation:
        summary_strat = SummaryExplanationStrategy()
        trace_strat = ExecutionTraceStrategy()
        breakdown_strat = StageBreakdownStrategy()

        summary_expl = summary_strat.generate_explanation(input_data, definition)
        trace_expl = trace_strat.generate_explanation(input_data, definition)
        breakdown_expl = breakdown_strat.generate_explanation(input_data, definition)

        merged_meta = dict(summary_expl.metadata)
        merged_meta.update(trace_expl.metadata)
        merged_meta.update(breakdown_expl.metadata)

        trace_ids = []
        if "trace_id" in summary_expl.metadata:
            trace_ids.append(summary_expl.metadata["trace_id"])
        if trace_expl.decision_trace:
            trace_ids.append(trace_expl.decision_trace.trace_id)
        for se in breakdown_expl.stage_explanations:
            trace_ids.append(se.trace_id)

        composite_str = f"composite_{'_'.join(trace_ids)}"
        merged_meta["trace_id"] = generate_sha256_trace_id(composite_str)

        return PipelineExecutionExplanation(
            execution_id=summary_expl.execution_id,
            pipeline_id=summary_expl.pipeline_id,
            claim_length=summary_expl.claim_length,
            success=summary_expl.success,
            total_latency_ms=summary_expl.total_latency_ms,
            configuration_fingerprint=summary_expl.configuration_fingerprint,
            schema_version=summary_expl.schema_version,
            execution_environment=summary_expl.execution_environment,
            summary=summary_expl.summary,
            stage_explanations=breakdown_expl.stage_explanations,
            decision_trace=trace_expl.decision_trace,
            telemetry_context=breakdown_expl.telemetry_context,
            metadata=merged_meta,
        )
