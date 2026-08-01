"""Severity policy strategies and base protocol interface (M3.5)."""

from typing import Protocol, runtime_checkable

from src.core.exceptions import FailureAnalysisConfigurationError
from src.core.failure.failure_models import (
    FailureSeverity,
    RootCauseResult,
    SeverityEvaluationResult,
    SeverityPolicyDefinition,
)

# Severity ordering for aggregation comparison (higher = more severe).
_SEVERITY_ORDER: dict[FailureSeverity, int] = {
    FailureSeverity.INFO: 1,
    FailureSeverity.LOW: 2,
    FailureSeverity.MEDIUM: 3,
    FailureSeverity.HIGH: 4,
    FailureSeverity.CRITICAL: 5,
}


@runtime_checkable
class BaseSeverityPolicy(Protocol):
    """Protocol for stateless severity policy evaluation over a root cause result."""

    def validate_compatibility(self, definition: SeverityPolicyDefinition) -> None:
        """Validates compatibility of the policy with the given definition."""
        ...

    def evaluate(
        self,
        root_cause_result: RootCauseResult,
        definition: SeverityPolicyDefinition,
    ) -> SeverityEvaluationResult:
        """Evaluates ordered SeverityRules to produce an immutable SeverityEvaluationResult."""
        ...


class ThresholdSeverityPolicy(BaseSeverityPolicy):
    """Stateless severity policy evaluating ordered SeverityRules.

    No hardcoded category precedence or thresholds.
    """

    def validate_compatibility(self, definition: SeverityPolicyDefinition) -> None:
        if not isinstance(definition, SeverityPolicyDefinition):
            raise FailureAnalysisConfigurationError(
                "ThresholdSeverityPolicy requires SeverityPolicyDefinition."
            )

    def evaluate(
        self,
        root_cause_result: RootCauseResult,
        definition: SeverityPolicyDefinition,
    ) -> SeverityEvaluationResult:
        confidence = root_cause_result.attribution_confidence

        # 1. Sort rules by priority (ascending = highest priority first).
        sorted_rules = sorted(definition.rules, key=lambda r: r.priority)

        applied_rule: str | None = None
        matched_severity = definition.default_severity
        escalation_required = False
        escalation_reason = ""
        policy_trace: list[str] = []

        # 2. Evaluate rules in priority order.
        for rule in sorted_rules:
            policy_trace.append(f"evaluated:{rule.rule_id}")

            # Check if confidence meets the rule minimum.
            if confidence < rule.minimum_confidence:
                policy_trace.append(
                    f"skipped:{rule.rule_id}:confidence_below_threshold"
                )
                continue

            # Check for category override on the primary root cause node.
            # Since we store node IDs (not categories) in the result, we match
            # by checking if a category override exists for this rule's category value.
            category_str = rule.category.value
            if category_str in definition.category_overrides:
                # Override takes precedence over rule severity.
                matched_severity = definition.category_overrides[category_str]
                escalation_required = rule.escalation_required
                escalation_reason = f"category_override:{category_str}"
                applied_rule = rule.rule_id
                policy_trace.append(f"applied_override:{rule.rule_id}")
                break

            # Apply rule severity.
            matched_severity = rule.severity
            escalation_required = rule.escalation_required
            if rule.escalation_required:
                escalation_reason = f"rule:{rule.rule_id}:escalation_required"
            applied_rule = rule.rule_id
            policy_trace.append(f"applied:{rule.rule_id}")
            break

        # 3. Collect contributing severities from dependency path nodes.
        contributing_severities: list[FailureSeverity] = [matched_severity]

        # 4. Overall severity = matched_severity (single primary evaluation).
        overall_severity = matched_severity

        return SeverityEvaluationResult(
            overall_severity=overall_severity,
            contributing_severities=tuple(contributing_severities),
            escalation_required=escalation_required,
            escalation_reason=escalation_reason,
            applied_rule=applied_rule,
            policy_trace=tuple(policy_trace),
        )
