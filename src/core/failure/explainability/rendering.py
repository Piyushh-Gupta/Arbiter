"""Failure report rendering component (M3.7)."""

import json

from src.core.failure.explainability.explanation_models import (
    FailureExplanationResult,
    FailureExplanationTemplate,
)


class FailureReportRenderer:
    """Stateless report renderer producing Markdown, JSON, and Plain Text representations."""

    def render_markdown(
        self,
        result: FailureExplanationResult,
        template: FailureExplanationTemplate | None = None,
    ) -> str:
        """Renders the explanation result as formatted Markdown."""
        lines: list[str] = [
            "# Failure Analysis Explanation Report",
            "",
            "## Summary",
            result.summary,
            "",
        ]

        if result.detailed_explanation:
            lines.extend(
                [
                    "## Detailed Explanation",
                    result.detailed_explanation,
                    "",
                ]
            )

        lines.extend(
            [
                "## Evidence Explanation",
                f"- **Supporting Diagnostics**: {', '.join(result.evidence_explanation.supporting_diagnostics) if result.evidence_explanation.supporting_diagnostics else 'None'}",
                f"- **Contributing Failures**: {', '.join(result.evidence_explanation.contributing_failures) if result.evidence_explanation.contributing_failures else 'None'}",
                f"- **Ignored Failures**: {', '.join(result.evidence_explanation.ignored_failures) if result.evidence_explanation.ignored_failures else 'None'}",
                f"- **Evidence Trace**: {', '.join(result.evidence_explanation.evidence_trace) if result.evidence_explanation.evidence_trace else 'None'}",
                "",
                "## Decision Trace",
                f"- **Correlation Path**: {' -> '.join(result.decision_trace.correlation_path) if result.decision_trace.correlation_path else 'None'}",
                f"- **Attribution Path**: {' -> '.join(result.decision_trace.attribution_path) if result.decision_trace.attribution_path else 'None'}",
                f"- **Severity Policy Path**: {' -> '.join(result.decision_trace.severity_policy_path) if result.decision_trace.severity_policy_path else 'None'}",
                "- **Reasoning Chain**:",
            ]
        )
        for step in result.decision_trace.reasoning_chain:
            lines.append(f"  - {step}")

        lines.extend(
            [
                "",
                "## Metadata",
                f"- **Strategy ID**: {result.metadata.strategy_id}",
                f"- **Configuration Fingerprint**: {result.metadata.configuration_fingerprint}",
                f"- **Schema Version**: {result.metadata.schema_version}",
                f"- **Generation Timestamp**: {result.metadata.generation_timestamp}",
            ]
        )

        return "\n".join(lines)

    def render_json(self, result: FailureExplanationResult) -> str:
        """Renders the explanation result as formatted JSON string."""
        return json.dumps(result.model_dump(), indent=2)

    def render_plain_text(
        self,
        result: FailureExplanationResult,
        template: FailureExplanationTemplate | None = None,
    ) -> str:
        """Renders the explanation result as plain text."""
        if template:
            sum_str = template.summary_template.format(summary=result.summary)
            det_str = template.detail_template.format(
                details=result.detailed_explanation
            )
            return f"{sum_str}\n{det_str}"

        lines: list[str] = [
            "FAILURE ANALYSIS EXPLANATION REPORT",
            "==================================",
            f"SUMMARY: {result.summary}",
            "",
        ]
        if result.detailed_explanation:
            lines.extend(
                [
                    "DETAILED EXPLANATION:",
                    result.detailed_explanation,
                    "",
                ]
            )

        lines.extend(
            [
                "EVIDENCE EXPLANATION:",
                f"  Supporting Diagnostics: {', '.join(result.evidence_explanation.supporting_diagnostics)}",
                f"  Contributing Failures: {', '.join(result.evidence_explanation.contributing_failures)}",
                "",
                "DECISION TRACE:",
                "  Reasoning Chain:",
            ]
        )
        for step in result.decision_trace.reasoning_chain:
            lines.append(f"    * {step}")

        lines.extend(
            [
                "",
                "METADATA:",
                f"  Strategy: {result.metadata.strategy_id}",
                f"  Fingerprint: {result.metadata.configuration_fingerprint}",
            ]
        )

        return "\n".join(lines)
