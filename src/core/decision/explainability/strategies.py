"""Concrete decision explanation strategies (M4.6)."""

import hashlib
from typing import Any

from src.core.decision.decision_models import DecisionExecutionContext
from src.core.decision.explainability.base import BaseDecisionExplanationStrategy
from src.core.decision.explainability.explainability_models import (
    DecisionExplanation,
    DecisionExplanationDefinition,
)


class SummaryExplanationStrategy(BaseDecisionExplanationStrategy):
    """Produces structured summary of a decision outcome."""

    @property
    def strategy_id(self) -> str:
        return "summary"

    def validate_compatibility(self, definition: DecisionExplanationDefinition) -> None:
        pass

    def generate_explanation(
        self,
        context: DecisionExecutionContext,
        definition: DecisionExplanationDefinition,
    ) -> DecisionExplanation:
        action = context.selected_action

        confidence = 0.5
        uncertainty = 0.5
        if context.risk_evaluation:
            confidence = getattr(context.risk_evaluation, "adjusted_confidence", 0.5)
            uncertainty = getattr(context.risk_evaluation, "adjusted_uncertainty", 0.5)
        elif context.decision_metrics:
            confidence = getattr(context.decision_metrics, "confidence", 0.5)
            uncertainty = getattr(context.decision_metrics, "uncertainty", 0.5)

        summary_dict = {
            "selected_action": action,
            "confidence": float(confidence),
            "uncertainty": float(uncertainty),
            "profile": getattr(context.execution_metadata, "profile", "default"),
        }

        # Generate a deterministic trace identifier for summary
        summary_repr = f"{action}_{confidence:.4f}_{uncertainty:.4f}"
        trace_id = hashlib.sha256(summary_repr.encode()).hexdigest()[:16]

        return DecisionExplanation(
            summary=summary_dict,
            rule_trace=(),
            risk_trace=(),
            decision_trace={},
            metadata={"trace_id": trace_id},
        )


class TraceAuditExplanationStrategy(BaseDecisionExplanationStrategy):
    """Produces structured rule and risk traces with deterministic trace identifiers."""

    @property
    def strategy_id(self) -> str:
        return "trace_audit"

    def validate_compatibility(self, definition: DecisionExplanationDefinition) -> None:
        pass

    def generate_explanation(
        self,
        context: DecisionExecutionContext,
        definition: DecisionExplanationDefinition,
    ) -> DecisionExplanation:
        # 1. Rule Traces
        rule_traces: list[dict[str, Any]] = []
        for r_eval in context.ordered_rule_evaluations:
            r_id = r_eval.rule_id
            matched = r_eval.matched
            priority = r_eval.priority

            # Deterministic trace ID for rule evaluation
            r_repr = f"rule_{r_id}_{matched}_{priority}"
            trace_id = hashlib.sha256(r_repr.encode()).hexdigest()[:16]

            rule_traces.append(
                {
                    "rule_id": r_id,
                    "matched": matched,
                    "priority": priority,
                    "explanation": r_eval.explanation,
                    "trace_id": trace_id,
                }
            )

        # 2. Risk Traces
        risk_traces: list[dict[str, Any]] = []
        if context.risk_evaluation and hasattr(context.risk_evaluation, "risk_traces"):
            r_eval_obj = context.risk_evaluation
            for r_trace in r_eval_obj.risk_traces:
                factor_id = r_trace.factor_id
                reason = r_trace.adjustment_reason
                conf_d = r_trace.confidence_delta
                unc_d = r_trace.uncertainty_delta

                # Deterministic trace ID for risk adjustment
                risk_repr = f"risk_{factor_id}_{conf_d:.4f}_{unc_d:.4f}"
                trace_id = hashlib.sha256(risk_repr.encode()).hexdigest()[:16]

                risk_traces.append(
                    {
                        "factor_id": factor_id,
                        "adjustment_reason": reason,
                        "confidence_delta": conf_d,
                        "uncertainty_delta": unc_d,
                        "trace_id": trace_id,
                    }
                )

        # 3. Decision Trace summary
        evaluated_rules = [r["rule_id"] for r in rule_traces]
        rejected_rules = [r["rule_id"] for r in rule_traces if not r["matched"]]
        selected_rules = [r["rule_id"] for r in rule_traces if r["matched"]]
        selected_rule = selected_rules[0] if selected_rules else None

        # Deterministic decision trace ID
        d_repr = f"decision_{selected_rule}_{len(evaluated_rules)}"
        decision_trace_id = hashlib.sha256(d_repr.encode()).hexdigest()[:16]

        decision_trace_dict = {
            "evaluated_rules": tuple(evaluated_rules),
            "rejected_rules": tuple(rejected_rules),
            "selected_rule": selected_rule,
            "trace_id": decision_trace_id,
        }

        return DecisionExplanation(
            summary={},
            rule_trace=tuple(rule_traces),
            risk_trace=tuple(risk_traces),
            decision_trace=decision_trace_dict,
            metadata={"trace_id": decision_trace_id},
        )


class CompositeExplanationStrategy(BaseDecisionExplanationStrategy):
    """Combines summary, rule, and risk traces into a single unified structured explanation."""

    @property
    def strategy_id(self) -> str:
        return "composite"

    def validate_compatibility(self, definition: DecisionExplanationDefinition) -> None:
        pass

    def generate_explanation(
        self,
        context: DecisionExecutionContext,
        definition: DecisionExplanationDefinition,
    ) -> DecisionExplanation:
        summary_strat = SummaryExplanationStrategy()
        trace_strat = TraceAuditExplanationStrategy()

        summary_expl = summary_strat.generate_explanation(context, definition)
        trace_expl = trace_strat.generate_explanation(context, definition)

        # Merge metadata and trace IDs
        combined_metadata = dict(summary_expl.metadata)
        combined_metadata.update(trace_expl.metadata)

        composite_repr = f"composite_{summary_expl.metadata.get('trace_id')}_{trace_expl.metadata.get('trace_id')}"
        combined_metadata["trace_id"] = hashlib.sha256(
            composite_repr.encode()
        ).hexdigest()[:16]

        return DecisionExplanation(
            summary=summary_expl.summary,
            rule_trace=trace_expl.rule_trace,
            risk_trace=trace_expl.risk_trace,
            decision_trace=trace_expl.decision_trace,
            metadata=combined_metadata,
        )
