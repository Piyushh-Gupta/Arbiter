"""Concrete decision report renderers (M4.6)."""

import json
from typing import Protocol, runtime_checkable

from src.core.decision.explainability.explainability_models import DecisionExplanation


@runtime_checkable
class BaseDecisionRenderer(Protocol):
    """Stateless protocol for decision explanation renderers."""

    @property
    def renderer_id(self) -> str:
        """Unique identifier for the renderer format."""
        ...

    def render(self, explanation: DecisionExplanation) -> str:
        """Renders structured DecisionExplanation into target format."""
        ...


class MarkdownDecisionRenderer(BaseDecisionRenderer):
    """Formats structured DecisionExplanation into a clean Markdown audit report."""

    @property
    def renderer_id(self) -> str:
        return "markdown"

    def render(self, explanation: DecisionExplanation) -> str:
        lines: list[str] = ["# Decision Audit Explanation Report", ""]

        # Summary Section
        if explanation.summary:
            lines.append("## Decision Summary")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|---|---|")
            for k, v in sorted(explanation.summary.items()):
                lines.append(f"| {k} | {v} |")
            lines.append("")

        # Rule Trace Section
        if explanation.rule_trace:
            lines.append("## Evaluated Decision Rules")
            lines.append("")
            lines.append("| Rule ID | Priority | Matched | Explanation | Trace ID |")
            lines.append("|---|---|---|---|---|")
            for rule in explanation.rule_trace:
                r_id = rule.get("rule_id", "N/A")
                priority = rule.get("priority", 0)
                matched = rule.get("matched", False)
                expl = rule.get("explanation", "")
                t_id = rule.get("trace_id", "N/A")
                lines.append(f"| {r_id} | {priority} | {matched} | {expl} | `{t_id}` |")
            lines.append("")

        # Risk Trace Section
        if explanation.risk_trace:
            lines.append("## Operational Risk Adjustments")
            lines.append("")
            lines.append(
                "| Factor ID | Reason | Confidence Delta | Uncertainty Delta | Trace ID |"
            )
            lines.append("|---|---|---|---|---|")
            for risk in explanation.risk_trace:
                f_id = risk.get("factor_id", "N/A")
                reason = risk.get("adjustment_reason", "")
                conf_d = risk.get("confidence_delta", 0.0)
                unc_d = risk.get("uncertainty_delta", 0.0)
                t_id = risk.get("trace_id", "N/A")
                lines.append(
                    f"| {f_id} | {reason} | {conf_d:+.4f} | {unc_d:+.4f} | `{t_id}` |"
                )
            lines.append("")

        # Decision Trace Section
        if explanation.decision_trace:
            lines.append("## Decision Execution Trace")
            lines.append("")
            d_trace = explanation.decision_trace
            lines.append(f"- **Selected Rule**: `{d_trace.get('selected_rule')}`")
            lines.append(
                f"- **Evaluated Rules**: {list(d_trace.get('evaluated_rules', ())) }"
            )
            lines.append(
                f"- **Rejected Rules**: {list(d_trace.get('rejected_rules', ())) }"
            )
            lines.append(f"- **Trace ID**: `{d_trace.get('trace_id')}`")
            lines.append("")

        # Metadata Section
        lines.append("## Audit Metadata")
        lines.append("")
        for k, v in sorted(explanation.metadata.items()):
            lines.append(f"- **{k}**: {v}")
        lines.append("")

        rendered = "\n".join(lines)
        return rendered


class JsonDecisionRenderer(BaseDecisionRenderer):
    """Formats structured DecisionExplanation into JSON string."""

    @property
    def renderer_id(self) -> str:
        return "json"

    def render(self, explanation: DecisionExplanation) -> str:
        data = {
            "summary": explanation.summary,
            "rule_trace": explanation.rule_trace,
            "risk_trace": explanation.risk_trace,
            "decision_trace": explanation.decision_trace,
            "metadata": explanation.metadata,
        }
        return json.dumps(data, indent=2, sort_keys=True)


class TextDecisionRenderer(BaseDecisionRenderer):
    """Formats structured DecisionExplanation into structured plain text."""

    @property
    def renderer_id(self) -> str:
        return "text"

    def render(self, explanation: DecisionExplanation) -> str:
        parts: list[str] = ["DECISION AUDIT REPORT", "====================="]

        # Summary
        if explanation.summary:
            parts.append("\nSUMMARY:")
            for k, v in sorted(explanation.summary.items()):
                parts.append(f"  {k}: {v}")

        # Rules
        if explanation.rule_trace:
            parts.append("\nRULES EVALUATED:")
            for rule in explanation.rule_trace:
                parts.append(
                    f"  Rule: {rule.get('rule_id')} | Matched: {rule.get('matched')} | Priority: {rule.get('priority')} | Trace ID: {rule.get('trace_id')}"
                )

        # Risk
        if explanation.risk_trace:
            parts.append("\nRISK ADJUSTMENTS:")
            for risk in explanation.risk_trace:
                parts.append(
                    f"  Factor: {risk.get('factor_id')} | Delta: {risk.get('confidence_delta'):+.3f}/{risk.get('uncertainty_delta'):+.3f} | Reason: {risk.get('adjustment_reason')} | Trace ID: {risk.get('trace_id')}"
                )

        # Trace
        if explanation.decision_trace:
            parts.append("\nEXECUTION TRACE:")
            d_trace = explanation.decision_trace
            parts.append(f"  Selected Rule: {d_trace.get('selected_rule')}")
            parts.append(
                f"  Evaluated Count: {len(d_trace.get('evaluated_rules', ())) }"
            )
            parts.append(f"  Trace ID: {d_trace.get('trace_id')}")

        parts.append("\nAUDIT METADATA:")
        for k, v in sorted(explanation.metadata.items()):
            parts.append(f"  {k}: {v}")

        rendered = "\n".join(parts)
        return rendered
