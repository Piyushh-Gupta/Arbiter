"""Policy engine and strategy implementations for Decision Engine Architecture Modernization (M4.1 & M4.2)."""

import hashlib
import time
from datetime import datetime, timezone
from typing import Any, Sequence

from src.core.decision.base import BaseDecisionPolicyEngine, BaseDecisionStrategy
from src.core.decision.decision_models import (
    DecisionAction,
    DecisionContext,
    DecisionDefinition,
    DecisionEngineMetadata,
    DecisionExecutionContext,
    DecisionExecutionMetadata,
    DecisionInput,
    DecisionMetadata,
    DecisionPolicyGroup,
    DecisionPolicyResult,
    DecisionResult,
    DecisionRule,
    DecisionRuleEvaluation,
    DecisionRuntimeMetadata,
    DecisionTrace,
    ThresholdDecisionDefinition,
    compute_decision_fingerprint,
)
from src.core.exceptions import DecisionConfigurationError


class DecisionPolicyEngine(BaseDecisionPolicyEngine):
    """
    Stateless engine evaluating immutable decision policy groups and rules according to precedence
    and generating immutable DecisionExecutionContext outputs.
    """

    def validate_compatibility(self, definition: Any) -> None:
        """Validates that the provided definition is compatible."""
        if not isinstance(definition, DecisionDefinition):
            raise DecisionConfigurationError(
                "Invalid definition type for DecisionPolicyEngine."
            )

    def evaluate(
        self,
        input_data: DecisionInput,
        policy_groups: Sequence[DecisionPolicyGroup] | None = None,
    ) -> DecisionExecutionContext:
        """
        Evaluates active decision policy groups and rules sorted by priority (descending),
        returning an immutable DecisionExecutionContext.
        """
        start_time = time.perf_counter()
        effective_groups = (
            policy_groups or PolicyDecisionStrategy.default_policy_groups()
        )
        active_groups = sorted(
            [g for g in effective_groups if g.enabled],
            key=lambda g: g.priority,
            reverse=True,
        )

        context = input_data.context
        definition = input_data.definition

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

        ordered_policy_results: list[DecisionPolicyResult] = []
        ordered_rule_evaluations: list[DecisionRuleEvaluation] = []
        selected_action: str | None = None

        total_rules_evaluated = 0

        for group in active_groups:
            matched_rules: list[str] = []
            reasoning: list[str] = []
            group_action: str | None = None
            escalation_req = False

            active_rules = sorted(
                [r for r in group.ordered_rules if r.enabled],
                key=lambda r: r.priority,
                reverse=True,
            )

            for rule in active_rules:
                total_rules_evaluated += 1
                conds = rule.conditions
                matched = True

                if "min_confidence" in conds and confidence < conds["min_confidence"]:
                    matched = False
                if (
                    "max_uncertainty" in conds
                    and uncertainty > conds["max_uncertainty"]
                ):
                    matched = False
                if (
                    "require_escalation" in conds
                    and conds["require_escalation"] != sev_escalation
                ):
                    matched = False

                rule_eval = DecisionRuleEvaluation(
                    rule_id=rule.rule_id,
                    matched=matched,
                    confidence_delta=0.0,
                    uncertainty_delta=0.0,
                    explanation=f"Rule {rule.rule_id} evaluation matched={matched}.",
                    priority=rule.priority,
                )
                ordered_rule_evaluations.append(rule_eval)

                if matched:
                    matched_rules.append(rule.rule_id)
                    reasoning.append(
                        f"Matched rule {rule.rule_id} in group {group.group_id} -> {rule.action}"
                    )
                    if group_action is None:
                        group_action = rule.action
                    if rule.action == "ESCALATE":
                        escalation_req = True
                    if selected_action is None:
                        selected_action = rule.action

            group_res = DecisionPolicyResult(
                group_id=group.group_id,
                matched_rules=tuple(matched_rules),
                confidence_delta=0.0,
                uncertainty_delta=0.0,
                escalation_requested=escalation_req,
                selected_action=group_action,
                reasoning=tuple(reasoning),
            )
            ordered_policy_results.append(group_res)

        final_action = selected_action or "ABSTAIN"
        duration_ms = (time.perf_counter() - start_time) * 1000.0

        fingerprint = compute_decision_fingerprint(definition)
        now_iso = datetime.now(timezone.utc).isoformat()

        runtime_metadata = DecisionRuntimeMetadata(
            policy_engine="DecisionPolicyEngine",
            configuration_fingerprint=fingerprint,
            schema_version="1.0",
            execution_timestamp=now_iso,
            execution_environment="production",
        )

        request_id = str(context.metadata.get("request_id", "req_default"))
        execution_metadata = DecisionExecutionMetadata(
            request_id=request_id,
            execution_duration_ms=duration_ms,
            profile=definition.decision_strategy,
            decision_policy=definition.confidence_policy,
        )

        exec_fp = hashlib.sha256(
            f"{fingerprint}_{len(ordered_rule_evaluations)}_{final_action}".encode()
        ).hexdigest()

        engine_metadata = DecisionEngineMetadata(
            engine_version="1.0",
            policy_engine_version="1.0",
            evaluated_group_count=len(active_groups),
            evaluated_rule_count=total_rules_evaluated,
            configuration_fingerprint=fingerprint,
            execution_fingerprint=exec_fp,
        )

        return DecisionExecutionContext(
            ordered_policy_results=tuple(ordered_policy_results),
            ordered_rule_evaluations=tuple(ordered_rule_evaluations),
            runtime_metadata=runtime_metadata,
            execution_metadata=execution_metadata,
            engine_metadata=engine_metadata,
            selected_action=final_action,
        )


class PolicyDecisionStrategy(BaseDecisionStrategy):
    """
    Stateless decision strategy orchestrating DecisionPolicyEngine to produce immutable DecisionResult outputs.
    """

    def __init__(
        self,
        policy_groups: Sequence[DecisionPolicyGroup] | None = None,
        policy_engine: BaseDecisionPolicyEngine | None = None,
        rules: Sequence[DecisionRule] | None = None,
    ) -> None:
        self.policy_engine = policy_engine or DecisionPolicyEngine()
        if policy_groups is not None:
            self.policy_groups = tuple(policy_groups)
        elif rules is not None:
            self.policy_groups = (
                DecisionPolicyGroup(
                    group_id="custom_rules_group",
                    priority=100,
                    enabled=True,
                    ordered_rules=tuple(rules),
                ),
            )
        else:
            self.policy_groups = self.default_policy_groups()

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

    @staticmethod
    def default_policy_groups() -> tuple[DecisionPolicyGroup, ...]:
        """Provides canonical default decision policy groups."""
        rules = PolicyDecisionStrategy.default_rules()
        g1 = DecisionPolicyGroup(
            group_id="group_escalation",
            priority=100,
            enabled=True,
            ordered_rules=(rules[0],),
        )
        g2 = DecisionPolicyGroup(
            group_id="group_routing",
            priority=80,
            enabled=True,
            ordered_rules=(rules[1], rules[2]),
        )
        g3 = DecisionPolicyGroup(
            group_id="group_fallback",
            priority=10,
            enabled=True,
            ordered_rules=(rules[3],),
        )
        return (g1, g2, g3)

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

        input_data = DecisionInput(context=context, definition=effective_def_obj)
        exec_context = self.policy_engine.evaluate(input_data, self.policy_groups)

        action = exec_context.selected_action
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
        escalation_req = action == "ESCALATE"

        evaluated_rules: list[str] = []
        rejected_rules: list[str] = []
        selected_rule: str | None = None
        policy_path: list[str] = []
        escalation_reasons: list[str] = []

        for eval_rule in exec_context.ordered_rule_evaluations:
            evaluated_rules.append(eval_rule.rule_id)
            if eval_rule.matched:
                if selected_rule is None:
                    selected_rule = eval_rule.rule_id
                policy_path.append(f"Rule {eval_rule.rule_id} matched.")
            else:
                rejected_rules.append(eval_rule.rule_id)

        for group_res in exec_context.ordered_policy_results:
            if group_res.escalation_requested:
                escalation_reasons.extend(group_res.reasoning)

        trace = DecisionTrace(
            evaluated_rules=tuple(evaluated_rules),
            rejected_rules=tuple(rejected_rules),
            selected_rule=selected_rule,
            confidence_evolution=(confidence,),
            uncertainty_evolution=(uncertainty,),
            escalation_reasoning=tuple(escalation_reasons),
            policy_path=tuple(policy_path),
        )

        expl_ref = None
        if context.explanation_result and hasattr(
            context.explanation_result, "summary"
        ):
            expl_ref = getattr(context.explanation_result, "summary", None)

        metadata = DecisionMetadata(
            strategy_id="policy_decision_strategy",
            configuration_fingerprint=exec_context.runtime_metadata.configuration_fingerprint,
            schema_version="1.0",
            generation_timestamp=exec_context.runtime_metadata.execution_timestamp,
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
