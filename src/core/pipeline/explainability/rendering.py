"""Concrete renderers for Pipeline Explainability (M5.5)."""

import json

from src.core.pipeline.explainability.base import BasePipelineExplanationRenderer
from src.core.pipeline.explainability.explainability_models import (
    PipelineExecutionExplanation,
)


class MarkdownPipelineRenderer(BasePipelineExplanationRenderer):
    """Formats structured PipelineExecutionExplanation into a Markdown report."""

    @property
    def renderer_id(self) -> str:
        return "markdown"

    def render(self, explanation: PipelineExecutionExplanation) -> str:
        lines = ["# Pipeline Execution Audit Report", ""]

        lines.append("## Execution Summary")
        lines.append("")
        lines.append("| Field | Value |")
        lines.append("|---|---|")
        lines.append(f"| execution_id | {explanation.execution_id} |")
        lines.append(f"| pipeline_id | {explanation.pipeline_id} |")
        lines.append(f"| success | {explanation.success} |")
        lines.append(f"| total_latency_ms | {explanation.total_latency_ms:.2f} |")
        lines.append(f"| claim_length | {explanation.claim_length} |")
        lines.append(f"| outcome | {explanation.summary.outcome} |")
        lines.append(f"| stage_count | {explanation.summary.stage_count} |")
        lines.append(
            f"| configuration_fingerprint | {explanation.configuration_fingerprint} |"
        )
        lines.append(f"| schema_version | {explanation.schema_version} |")
        lines.append(f"| execution_environment | {explanation.execution_environment} |")
        lines.append("")

        if explanation.stage_explanations:
            lines.append("## Stage Latency Breakdown")
            lines.append("")
            lines.append(
                "| Stage ID | Profile | Latency (ms) | Success | Percentile Rank | Observation |"
            )
            lines.append("|---|---|---|---|---|---|")
            for se in explanation.stage_explanations:
                rank_str = (
                    f"{se.latency_percentile_rank:.1f}%"
                    if se.latency_percentile_rank is not None
                    else "N/A"
                )
                lines.append(
                    f"| {se.stage_id} | {se.profile_id} | {se.latency_ms:.2f} | "
                    f"{se.success} | {rank_str} | {se.observation} |"
                )
            lines.append("")

        if explanation.decision_trace is not None:
            dt = explanation.decision_trace
            lines.append("## Resilience Decision Trace")
            lines.append("")
            lines.append(f"- **Total Attempts**: {dt.total_attempts}")
            succeeded_val = (
                dt.succeeded_on_attempt
                if dt.succeeded_on_attempt is not None
                else "N/A"
            )
            lines.append(f"- **Succeeded On Attempt**: {succeeded_val}")
            lines.append(f"- **Timeout Enforced**: {dt.timeout_enforced}")
            lines.append(f"- **Recovery Invoked**: {dt.recovery_invoked}")
            rec_strat = (
                dt.recovery_strategy_id
                if dt.recovery_strategy_id is not None
                else "N/A"
            )
            lines.append(f"- **Recovery Strategy ID**: {rec_strat}")
            term_err = dt.terminal_error if dt.terminal_error is not None else "N/A"
            lines.append(f"- **Terminal Error**: {term_err}")
            lines.append(
                f"- **Total Retry Overhead**: {dt.total_retry_overhead_ms:.2f}ms"
            )
            lines.append(f"- **Trace ID**: `{dt.trace_id}`")
            lines.append("")

        if explanation.telemetry_context.total_executions is not None:
            tc = explanation.telemetry_context
            lines.append("## Telemetry Context")
            lines.append("")
            lines.append(f"- **Total Executions**: {tc.total_executions}")
            rate_val = (
                f"{tc.overall_success_rate:.2%}"
                if tc.overall_success_rate is not None
                else "N/A"
            )
            lines.append(f"- **Overall Success Rate**: {rate_val}")
            lines.append("")

        if explanation.metadata:
            lines.append("## Audit Metadata")
            lines.append("")
            for k, v in sorted(explanation.metadata.items()):
                lines.append(f"- **{k}**: {v}")
            lines.append("")

        return "\n".join(lines)


class JsonPipelineRenderer(BasePipelineExplanationRenderer):
    """Formats structured PipelineExecutionExplanation into a JSON string."""

    @property
    def renderer_id(self) -> str:
        return "json"

    def render(self, explanation: PipelineExecutionExplanation) -> str:
        return json.dumps(explanation.model_dump(), indent=2, sort_keys=True)


class TextPipelineRenderer(BasePipelineExplanationRenderer):
    """Formats structured PipelineExecutionExplanation into structured plain text."""

    @property
    def renderer_id(self) -> str:
        return "text"

    def render(self, explanation: PipelineExecutionExplanation) -> str:
        parts = ["PIPELINE EXECUTION AUDIT REPORT", "==============================="]

        parts.append(f"Execution ID: {explanation.execution_id}")
        parts.append(f"Pipeline ID: {explanation.pipeline_id}")
        parts.append(f"Success: {explanation.success}")
        parts.append(f"Total Latency: {explanation.total_latency_ms:.2f}ms")
        parts.append(f"Claim Length: {explanation.claim_length}")
        parts.append(f"Outcome: {explanation.summary.outcome}")
        parts.append(f"Stage Count: {explanation.summary.stage_count}")
        parts.append(
            f"Configuration Fingerprint: {explanation.configuration_fingerprint}"
        )
        parts.append(f"Schema Version: {explanation.schema_version}")
        parts.append(f"Execution Environment: {explanation.execution_environment}")

        if explanation.stage_explanations:
            parts.append("\nSTAGE BREAKDOWN:")
            for se in explanation.stage_explanations:
                rank_str = (
                    f"{se.latency_percentile_rank:.1f}%"
                    if se.latency_percentile_rank is not None
                    else "N/A"
                )
                parts.append(
                    f"  Stage: {se.stage_id} | Profile: {se.profile_id} | "
                    f"Latency: {se.latency_ms:.2f}ms | Success: {se.success} | "
                    f"Percentile Rank: {rank_str} | Obs: {se.observation}"
                )

        if explanation.decision_trace is not None:
            dt = explanation.decision_trace
            parts.append("\nRESILIENCE DECISION TRACE:")
            parts.append(f"  Total Attempts: {dt.total_attempts}")
            succeeded_val = (
                dt.succeeded_on_attempt
                if dt.succeeded_on_attempt is not None
                else "N/A"
            )
            parts.append(f"  Succeeded On Attempt: {succeeded_val}")
            parts.append(f"  Timeout Enforced: {dt.timeout_enforced}")
            parts.append(f"  Recovery Invoked: {dt.recovery_invoked}")
            rec_strat = (
                dt.recovery_strategy_id
                if dt.recovery_strategy_id is not None
                else "N/A"
            )
            parts.append(f"  Recovery Strategy ID: {rec_strat}")
            term_err = dt.terminal_error if dt.terminal_error is not None else "N/A"
            parts.append(f"  Terminal Error: {term_err}")
            parts.append(f"  Total Retry Overhead: {dt.total_retry_overhead_ms:.2f}ms")
            parts.append(f"  Trace ID: {dt.trace_id}")

        if explanation.telemetry_context.total_executions is not None:
            tc = explanation.telemetry_context
            parts.append("\nTELEMETRY CONTEXT:")
            parts.append(f"  Total Executions: {tc.total_executions}")
            rate_val = (
                f"{tc.overall_success_rate:.2%}"
                if tc.overall_success_rate is not None
                else "N/A"
            )
            parts.append(f"  Overall Success Rate: {rate_val}")

        if explanation.metadata:
            parts.append("\nAUDIT METADATA:")
            for k, v in sorted(explanation.metadata.items()):
                parts.append(f"  {k}: {v}")

        return "\n".join(parts)
