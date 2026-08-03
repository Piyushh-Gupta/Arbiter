"""Unit and integration tests for M4.4 Risk & Failure-Aware Decision Engine."""

import pytest

from src.core.bootstrap import build_decision_registry
from src.core.config import Settings
from src.core.decision import (
    DecisionContext,
    DecisionDefinition,
    DecisionInput,
    DecisionMetrics,
    DecisionPolicyEngine,
    DecisionProfileRegistry,
    DecisionResult,
    PolicyDecisionStrategy,
    RiskEvaluation,
    RiskPolicyRegistry,
    RiskTrace,
)
from src.core.decision.policies import (
    CostBenefitRiskPolicy,
    RawRiskPolicy,
    SeverityThresholdRiskPolicy,
)
from src.core.exceptions import (
    DecisionRiskPolicyNotFoundError,
    DuplicateDecisionRiskPolicyError,
)


# Mock classes for severity & failure analysis results
class MockSeverityResult:
    def __init__(
        self, overall_severity: str, escalation_required: bool = False
    ) -> None:
        self.overall_severity = overall_severity
        self.escalation_required = escalation_required


class MockFailureAnalysisResult:
    def __init__(self, failure_flags: tuple[str, ...]) -> None:
        self.failure_flags = failure_flags


class MockVerificationResult:
    def __init__(self, confidence: float) -> None:
        self.confidence = confidence


# --- Risk Registry Tests ---


def test_risk_registry_duplicate_ids_raises() -> None:
    p1 = RawRiskPolicy()
    p2 = RawRiskPolicy()  # Duplicate policy_id: "raw_risk"

    with pytest.raises(DuplicateDecisionRiskPolicyError) as exc_info:
        RiskPolicyRegistry(policies=(p1, p2))
    assert "Duplicate risk policy ID detected: raw_risk" in str(exc_info.value)


def test_risk_registry_lookup_success_and_failure() -> None:
    p_raw = RawRiskPolicy()
    p_sev = SeverityThresholdRiskPolicy()
    registry = RiskPolicyRegistry(policies=(p_raw, p_sev))

    # Successful resolve
    assert registry.resolve("raw_risk") is p_raw
    assert registry.resolve("severity_aware") is p_sev

    # Failed resolve
    with pytest.raises(DecisionRiskPolicyNotFoundError) as exc_info:
        registry.resolve("non_existent_policy")
    assert "Decision risk policy not found: non_existent_policy" in str(exc_info.value)


def test_risk_registry_compatibility_validation() -> None:
    p_raw = RawRiskPolicy()
    registry = RiskPolicyRegistry(policies=(p_raw,))

    # Compatible definition
    defn = DecisionDefinition(failure_policy="raw_risk")
    registry.validate_compatibility(defn)

    # Incompatible/Missing policy configuration
    defn_missing = DecisionDefinition(failure_policy="missing_risk_policy")
    with pytest.raises(DecisionRiskPolicyNotFoundError) as exc_info:
        registry.validate_compatibility(defn_missing)
    assert "Risk policy 'missing_risk_policy' not found in registry" in str(
        exc_info.value
    )


# --- Concrete Risk Policies & Trace Recording Tests ---


def test_raw_risk_policy_behavior() -> None:
    policy = RawRiskPolicy()
    assert policy.policy_id == "raw_risk"

    metrics = DecisionMetrics(
        confidence=0.8, uncertainty=0.2, calibrated=True, source="calibration"
    )
    context = DecisionContext()
    definition = DecisionDefinition()

    eval_result = policy.evaluate_risk(context, metrics, definition)

    assert isinstance(eval_result, RiskEvaluation)
    assert eval_result.risk_score == 0.2
    assert eval_result.adjusted_confidence == 0.8
    assert eval_result.adjusted_uncertainty == 0.2
    assert eval_result.applied_policy_id == "raw_risk"
    assert eval_result.contributing_factors == ("baseline",)

    assert len(eval_result.risk_traces) == 1
    trace = eval_result.risk_traces[0]
    assert isinstance(trace, RiskTrace)
    assert trace.factor_id == "baseline"
    assert "Baseline risk mapping" in trace.adjustment_reason
    assert trace.confidence_delta == 0.0
    assert trace.uncertainty_delta == 0.0


def test_severity_aware_policy_no_severity() -> None:
    policy = SeverityThresholdRiskPolicy()
    metrics = DecisionMetrics(
        confidence=0.8, uncertainty=0.2, calibrated=True, source="calibration"
    )
    context = DecisionContext()  # No severity result
    definition = DecisionDefinition()

    eval_result = policy.evaluate_risk(context, metrics, definition)
    assert eval_result.adjusted_confidence == 0.8
    assert eval_result.adjusted_uncertainty == 0.2
    assert len(eval_result.risk_traces) == 0


def test_severity_aware_policy_with_severity() -> None:
    policy = SeverityThresholdRiskPolicy()
    metrics = DecisionMetrics(
        confidence=0.8, uncertainty=0.2, calibrated=True, source="calibration"
    )

    # 1. Medium severity
    context_med = DecisionContext(
        severity_result=MockSeverityResult(overall_severity="MEDIUM")
    )
    eval_med = policy.evaluate_risk(context_med, metrics, DecisionDefinition())
    assert eval_med.adjusted_confidence == pytest.approx(0.65)  # 0.8 - 0.15
    assert eval_med.adjusted_uncertainty == pytest.approx(0.35)  # 0.2 + 0.15
    assert "severity_penalty" in eval_med.contributing_factors
    assert len(eval_med.risk_traces) == 1
    assert eval_med.risk_traces[0].confidence_delta == -0.15
    assert eval_med.risk_traces[0].uncertainty_delta == 0.15

    # 2. Critical severity + failure flags (composable independent factors)
    context_crit = DecisionContext(
        severity_result=MockSeverityResult(overall_severity="CRITICAL"),
        failure_analysis_result=MockFailureAnalysisResult(
            failure_flags=("err1", "err2")
        ),
    )
    eval_crit = policy.evaluate_risk(context_crit, metrics, DecisionDefinition())

    # Severity penalty for CRITICAL is 0.50. Flags penalty for 2 flags is 0.10. Total penalty = 0.60
    assert eval_crit.adjusted_confidence == pytest.approx(0.20)  # 0.8 - 0.6
    assert eval_crit.adjusted_uncertainty == pytest.approx(0.80)  # 0.2 + 0.6
    assert set(eval_crit.contributing_factors) == {
        "severity_penalty",
        "failure_flags_penalty",
    }
    assert len(eval_crit.risk_traces) == 2


def test_cost_benefit_policy_behavior() -> None:
    policy = CostBenefitRiskPolicy()

    # 1. Low uncertainty, no errors -> no adjustments
    metrics_low = DecisionMetrics(
        confidence=0.8, uncertainty=0.2, calibrated=True, source="calibration"
    )
    context_clean = DecisionContext()
    eval_clean = policy.evaluate_risk(context_clean, metrics_low, DecisionDefinition())
    assert eval_clean.adjusted_confidence == 0.8
    assert eval_clean.adjusted_uncertainty == 0.2
    assert len(eval_clean.risk_traces) == 0

    # 2. High uncertainty (>0.4) and errors -> both factors applied
    metrics_high = DecisionMetrics(
        confidence=0.5, uncertainty=0.5, calibrated=True, source="calibration"
    )
    context_err = DecisionContext(
        failure_analysis_result=MockFailureAnalysisResult(failure_flags=("flag1",))
    )
    eval_err = policy.evaluate_risk(context_err, metrics_high, DecisionDefinition())

    # Uncertainty risk penalty: 0.10. Error risk penalty: 0.20. Total = 0.30
    assert eval_err.adjusted_confidence == pytest.approx(0.20)  # 0.5 - 0.3
    assert eval_err.adjusted_uncertainty == pytest.approx(0.80)  # 0.5 + 0.3
    assert set(eval_err.contributing_factors) == {"uncertainty_risk", "error_risk"}
    assert len(eval_err.risk_traces) == 2


# --- Engine Integration & Rule Interaction Tests ---


def test_engine_resolves_and_evaluates_risk() -> None:
    # Set up engine with SeverityThresholdRiskPolicy
    engine = DecisionPolicyEngine()

    # Set up input with critical severity to trigger adjustment
    context = DecisionContext(
        severity_result=MockSeverityResult(overall_severity="CRITICAL"),
        verification_result=MockVerificationResult(confidence=0.9),
    )
    definition = DecisionDefinition(
        confidence_policy="raw", failure_policy="severity_aware"
    )
    input_data = DecisionInput(context=context, definition=definition)

    exec_ctx = engine.evaluate(input_data)

    assert exec_ctx.risk_evaluation is not None
    # Raw confidence was 0.9. Critical penalty is 0.5. Adjusted confidence = 0.4
    assert exec_ctx.risk_evaluation.adjusted_confidence == pytest.approx(0.4)
    assert exec_ctx.risk_evaluation.adjusted_uncertainty == pytest.approx(0.6)

    # Verify it updated the rule evaluation conditions (i.e. rules checking min_confidence 0.8 should fail)
    # The default rule "rule_accept_high_confidence" requires min_confidence=0.8 and max_uncertainty=0.3.
    # Since confidence is now adjusted to 0.4 and uncertainty to 0.6, it should match "rule_reject_low_confidence" (max_uncertainty 0.7).
    matched_rules: list[str] = []
    for pe_res in exec_ctx.ordered_policy_results:
        matched_rules.extend(pe_res.matched_rules)

    assert "rule_accept_high_confidence" not in matched_rules
    assert "rule_reject_low_confidence" in matched_rules


# --- Bootstrap Verification Tests ---


def test_bootstrap_builds_risk_registries_correctly() -> None:
    config = Settings()
    registry = build_decision_registry(config)

    assert isinstance(registry, DecisionProfileRegistry)
    profile = registry.resolve("default_decision")

    # Engine must have injected risk registry
    engine = profile.strategy.policy_engine
    assert isinstance(engine, DecisionPolicyEngine)
    assert isinstance(engine.risk_policy_registry, RiskPolicyRegistry)

    # The default profile definition has failure_policy = "severity_aware"
    assert profile.definition.failure_policy == "severity_aware"

    # Try resolving it
    policy = engine.risk_policy_registry.resolve("severity_aware")
    assert isinstance(policy, SeverityThresholdRiskPolicy)


# --- End-to-End Integration Pipeline Tests ---


def test_decision_risk_pipeline_integration() -> None:
    # Setup full context with raw confidence and critical severity failure
    context = DecisionContext(
        verification_result=MockVerificationResult(confidence=0.95),
        severity_result=MockSeverityResult(overall_severity="CRITICAL"),
    )
    definition = DecisionDefinition(
        confidence_policy="raw", failure_policy="severity_aware"
    )

    strategy = PolicyDecisionStrategy()
    result = strategy.decide(context, definition)

    assert isinstance(result, DecisionResult)
    # Raw confidence = 0.95. Penalty = 0.5. Adjusted = 0.45. Adjusted uncertainty = 0.55.
    assert result.final_confidence == pytest.approx(0.45)
    assert result.final_uncertainty == pytest.approx(0.55)
    assert (
        result.final_verdict == "REJECT"
    )  # Adjusted uncertainty 0.55 <= 0.70 default threshold
