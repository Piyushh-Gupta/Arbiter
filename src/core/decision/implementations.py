"""Policy engine and strategy implementations for Decision Engine Architecture Modernization (M4.1)."""

from datetime import datetime, timezone
from typing import Any, Sequence

from src.core.decision.base import BaseDecisionStrategy
from src.core.decision.decision_models import (
    DecisionAction,
    DecisionContext,
    DecisionDefinition,
    DecisionMetadata,
    DecisionResult,
    DecisionRule,
    DecisionTrace,
    ThresholdDecisionDefinition,
    compute_decision_fingerprint,
)
from src.core.exceptions import DecisionConfigurationError


class DecisionPolicyEngine:
    """
    Stateless engine evaluating immutable decision rules according to precedence
    and generating execution traces.
    """

    def evaluate(
        self,
        context: DecisionContext,
        rules: Sequence[DecisionRule],
    ) -> tuple[str, DecisionTrace]:
        """
        Evaluates active decision rules sorted by priority (descending) and returns
        the selected action and execution trace.
        """
        active_rules = sorted(
            [r for r in rules if r.enabled],
            key=lambda r: r.priority,
            reverse=True,
        )

        evaluated_rules: list[str] = []
        rejected_rules: list[str] = []
        escalation_reasons: list[str] = []
        policy_path: list[str] = []
        selected_rule: str | None = None
        action: str = "ABSTAIN"

        # Extract values from context
        confidence = 0.5
        if context.verification_result and hasattr(
            context.verification_result, "confidence"
        ):
            confidence = getattr(context.verification_result, "confidence", 0.5)
        elif context.calibration_result and hasattr(
            context.calibration_result, "calibrated_confidence"
        ):
            confidence = getattr(
                context.calibration_result, "calibrated_confidence", 0.5
            )

        uncertainty = 1.0 - confidence

        sev_escalation = False
        if context.severity_result and hasattr(
            context.severity_result, "escalation_required"
        ):
            sev_escalation = getattr(
                context.severity_result, "escalation_required", False
            )

        for rule in active_rules:
            evaluated_rules.append(rule.rule_id)
            conds = rule.conditions
            matched = True

            if "min_confidence" in conds and confidence < conds["min_confidence"]:
                matched = False
            if "max_uncertainty" in conds and uncertainty > conds["max_uncertainty"]:
                matched = False
            if (
                "require_escalation" in conds
                and conds["require_escalation"] != sev_escalation
            ):
                matched = False

            if matched:
                selected_rule = rule.rule_id
                action = rule.action
                policy_path.append(f"Matched rule {rule.rule_id} -> {rule.action}")
                if rule.action == "ESCALATE":
                    escalation_reasons.append(
                        f"Rule {rule.rule_id} triggered escalation."
                    )
                break
            else:
                rejected_rules.append(rule.rule_id)

        if not selected_rule:
            policy_path.append("No rule matched; fell back to default ABSTAIN.")

        trace = DecisionTrace(
            evaluated_rules=tuple(evaluated_rules),
            rejected_rules=tuple(rejected_rules),
            selected_rule=selected_rule,
            confidence_evolution=(confidence,),
            uncertainty_evolution=(uncertainty,),
            escalation_reasoning=tuple(escalation_reasons),
            policy_path=tuple(policy_path),
        )

        return action, trace


class PolicyDecisionStrategy(BaseDecisionStrategy):
    """
    Stateless decision strategy orchestrating DecisionPolicyEngine to produce immutable DecisionResult outputs.
    """

    def __init__(
        self,
        rules: Sequence[DecisionRule] | None = None,
        policy_engine: DecisionPolicyEngine | None = None,
    ) -> None:
        self.policy_engine = policy_engine or DecisionPolicyEngine()
        self.rules = tuple(rules) if rules is not None else self.default_rules()

    @staticmethod
    def default_rules() -> tuple[DecisionRule, ...]:
        """Provides canonical default decision rules."""
        r1 = DecisionRule(
            rule_id="rule_escalate_critical",
            priority=100,
            enabled=True,
            conditions={"require_escalation": True},
            action="ESCALATE",
        )
        r2 = DecisionRule(
            rule_id="rule_accept_high_confidence",
            priority=80,
            enabled=True,
            conditions={"min_confidence": 0.8, "max_uncertainty": 0.3},
            action="ACCEPT",
        )
        r3 = DecisionRule(
            rule_id="rule_reject_low_confidence",
            priority=70,
            enabled=True,
            conditions={"max_uncertainty": 0.7},
            action="REJECT",
        )
        r4 = DecisionRule(
            rule_id="rule_fallback_abstain",
            priority=1,
            enabled=True,
            conditions={},
            action="ABSTAIN",
        )
        return (r1, r2, r3, r4)

    def validate_compatibility(self, definition: Any) -> None:
        """Validates that the provided definition is compatible."""
        if not isinstance(definition, DecisionDefinition):
            raise DecisionConfigurationError(
                "Invalid definition type for PolicyDecisionStrategy."
            )

    def decide(
        self,
        context_or_claim: Any,
        definition_or_unc: Any = None,
        definition: Any = None,
    ) -> DecisionResult:
        """Evaluates decision policies over context and returns immutable DecisionResult."""
        if isinstance(context_or_claim, DecisionContext):
            context = context_or_claim
            effective_def = definition_or_unc or DecisionDefinition()
        else:
            context = DecisionContext(
                calibration_result=definition_or_unc,
                metadata={"claim": str(context_or_claim)},
            )
            effective_def = definition or DecisionDefinition()

        effective_def_obj = (
            effective_def
            if isinstance(effective_def, DecisionDefinition)
            else DecisionDefinition()
        )
        self.validate_compatibility(effective_def_obj)

        action, trace = self.policy_engine.evaluate(context, self.rules)

        confidence = (
            trace.confidence_evolution[0] if trace.confidence_evolution else 0.5
        )
        uncertainty = (
            trace.uncertainty_evolution[0] if trace.uncertainty_evolution else 0.5
        )
        escalation_req = action == "ESCALATE"

        expl_ref = None
        if context.explanation_result and hasattr(
            context.explanation_result, "summary"
        ):
            expl_ref = getattr(context.explanation_result, "summary", None)

        metadata = DecisionMetadata(
            strategy_id="policy_decision_strategy",
            configuration_fingerprint=compute_decision_fingerprint(effective_def_obj),
            schema_version="1.0",
            generation_timestamp=datetime.now(timezone.utc).isoformat(),
        )

        return DecisionResult(
            action=action,
            rationale=f"Policy decision evaluated action: {action}",
            uncertainty_result=context.calibration_result,
            final_verdict=action,
            final_confidence=confidence,
            final_uncertainty=uncertainty,
            escalation_required=escalation_req,
            explanation_reference=expl_ref,
            decision_trace=trace,
            metadata=metadata,
        )


class ThresholdDecisionEngine:
    """Threshold-based decision engine for backward compatibility."""

    def validate_compatibility(self, definition: DecisionDefinition) -> None:
        """Validates configuration parameters."""
        if not isinstance(definition, ThresholdDecisionDefinition):
            raise DecisionConfigurationError(
                "ThresholdDecisionEngine requires ThresholdDecisionDefinition."
            )

    def decide(
        self,
        claim_or_context: Any,
        uncertainty_result_or_def: Any = None,
        definition: Any = None,
    ) -> DecisionResult:
        """Executes threshold-based routing."""
        if isinstance(claim_or_context, DecisionContext):
            context = claim_or_context
            effective_def = uncertainty_result_or_def or ThresholdDecisionDefinition()
            unc_result = context.calibration_result
        else:
            unc_result = uncertainty_result_or_def
            effective_def = definition or ThresholdDecisionDefinition()

        self.validate_compatibility(effective_def)

        unc_score = getattr(unc_result, "score", 0.5) if unc_result else 0.5
        label = None
        if unc_result and hasattr(unc_result, "failure_analysis_result"):
            fa = getattr(unc_result, "failure_analysis_result", None)
            if fa and hasattr(fa, "verification_result"):
                vr = getattr(fa, "verification_result", None)
                if vr and hasattr(vr, "label"):
                    label = getattr(vr, "label", None)

        label_str = (
            str(label.value if hasattr(label, "value") else label)
            if label
            else "UNKNOWN"
        )
        action = DecisionAction.ESCALATE
        rationale = f"Uncertainty score ({unc_score}) exceeds acceptable thresholds or label ({label_str}) lacks deterministic routing logic."

        accept_max = getattr(effective_def, "accept_max_uncertainty", 0.3)
        reject_max = getattr(effective_def, "reject_max_uncertainty", 0.7)

        if label_str == "SUPPORTS" and unc_score <= accept_max:
            action = DecisionAction.ACCEPT
            rationale = f"Claim supported with uncertainty ({unc_score}) <= threshold ({accept_max})"
        elif label_str == "REFUTES" and unc_score <= reject_max:
            action = DecisionAction.REJECT
            rationale = f"Claim refuted with uncertainty ({unc_score}) <= threshold ({reject_max})"

        metadata = DecisionMetadata(
            strategy_id="threshold_decision_engine",
            configuration_fingerprint="legacy",
            generation_timestamp=datetime.now(timezone.utc).isoformat(),
        )

        return DecisionResult(
            action=action,
            rationale=rationale,
            uncertainty_result=unc_result,
            final_verdict=str(action.value if hasattr(action, "value") else action),
            final_confidence=1.0 - unc_score,
            final_uncertainty=unc_score,
            metadata=metadata,
        )
