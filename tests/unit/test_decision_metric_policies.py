"""Unit and integration tests for M4.3 Threshold & Confidence Policies."""

from typing import Any

import pytest

from src.core.bootstrap import build_decision_registry
from src.core.config import Settings
from src.core.decision import (
    DecisionContext,
    DecisionDefinition,
    DecisionInput,
    DecisionMetricPolicyRegistry,
    DecisionMetrics,
    DecisionPolicyEngine,
    DecisionProfileRegistry,
    DecisionResult,
    PolicyDecisionStrategy,
)
from src.core.decision.policies import (
    CalibratedMetricPolicy,
    DecisionMetricResolver,
    EntropyMetricPolicy,
    RawMetricPolicy,
)
from src.core.exceptions import (
    DecisionExecutionError,
    DecisionMetricPolicyNotFoundError,
    DuplicateDecisionMetricPolicyError,
)


# Mock classes for verification & calibration results
class MockVerificationResult:
    def __init__(self, confidence: float) -> None:
        self.confidence = confidence


class MockCalibrationResult:
    def __init__(self, calibrated_confidence: float) -> None:
        self.calibrated_confidence = calibrated_confidence


# --- Base Decision Metric Policy Registry Tests ---


def test_registry_duplicate_ids_raises() -> None:
    p1 = RawMetricPolicy()
    p2 = RawMetricPolicy()  # Duplicate policy_id: "raw"

    with pytest.raises(DuplicateDecisionMetricPolicyError) as exc_info:
        DecisionMetricPolicyRegistry(policies=(p1, p2))
    assert "Duplicate metric policy ID detected: raw" in str(exc_info.value)


def test_registry_lookup_success_and_failure() -> None:
    p_raw = RawMetricPolicy()
    p_cal = CalibratedMetricPolicy()
    registry = DecisionMetricPolicyRegistry(policies=(p_raw, p_cal))

    # Successful resolve
    assert registry.resolve("raw") is p_raw
    assert registry.resolve("calibrated") is p_cal

    # Failed resolve
    with pytest.raises(DecisionMetricPolicyNotFoundError) as exc_info:
        registry.resolve("entropy")
    assert "Decision metric policy not found: entropy" in str(exc_info.value)


def test_registry_compatibility_validation() -> None:
    p_raw = RawMetricPolicy()
    registry = DecisionMetricPolicyRegistry(policies=(p_raw,))

    # Compatible definition
    defn = DecisionDefinition(confidence_policy="raw")
    registry.validate_compatibility(defn)

    # Incompatible/Missing policy configuration
    defn_missing = DecisionDefinition(confidence_policy="missing_policy")
    with pytest.raises(DecisionMetricPolicyNotFoundError) as exc_info:
        registry.validate_compatibility(defn_missing)
    assert "Metric policy 'missing_policy' not found in registry" in str(exc_info.value)


# --- Concrete Metric Policies Tests ---


def test_calibrated_metric_policy_execution() -> None:
    policy = CalibratedMetricPolicy()
    assert policy.policy_id == "calibrated"

    definition = DecisionDefinition()

    # Missing calibration result raises error
    context_missing = DecisionContext()
    with pytest.raises(DecisionExecutionError) as exc_info:
        policy.evaluate_metrics(context_missing, definition)
    assert "Calibration result is missing" in str(exc_info.value)

    # Missing calibrated_confidence raises error
    context_no_conf = DecisionContext(calibration_result=object())
    with pytest.raises(DecisionExecutionError) as exc_info:
        policy.evaluate_metrics(context_no_conf, definition)
    assert "Calibrated confidence score is missing" in str(exc_info.value)

    # Success path
    cal_res = MockCalibrationResult(calibrated_confidence=0.85)
    context_success = DecisionContext(calibration_result=cal_res)
    metrics = policy.evaluate_metrics(context_success, definition)

    assert isinstance(metrics, DecisionMetrics)
    assert metrics.confidence == 0.85
    assert metrics.uncertainty == pytest.approx(0.15)
    assert metrics.calibrated is True
    assert metrics.source == "calibration"
    assert metrics.metadata == {"policy_id": "calibrated"}


def test_raw_metric_policy_execution() -> None:
    policy = RawMetricPolicy()
    assert policy.policy_id == "raw"

    definition = DecisionDefinition()

    # Missing verification result raises error
    context_missing = DecisionContext()
    with pytest.raises(DecisionExecutionError) as exc_info:
        policy.evaluate_metrics(context_missing, definition)
    assert "Verification result is missing" in str(exc_info.value)

    # Missing confidence raises error
    context_no_conf = DecisionContext(verification_result=object())
    with pytest.raises(DecisionExecutionError) as exc_info:
        policy.evaluate_metrics(context_no_conf, definition)
    assert "Confidence score is missing" in str(exc_info.value)

    # Success path
    ver_res = MockVerificationResult(confidence=0.75)
    context_success = DecisionContext(verification_result=ver_res)
    metrics = policy.evaluate_metrics(context_success, definition)

    assert isinstance(metrics, DecisionMetrics)
    assert metrics.confidence == 0.75
    assert metrics.uncertainty == pytest.approx(0.25)
    assert metrics.calibrated is False
    assert metrics.source == "verification"
    assert metrics.metadata == {"policy_id": "raw"}


def test_entropy_metric_policy_execution() -> None:
    policy = EntropyMetricPolicy()
    assert policy.policy_id == "entropy"

    definition = DecisionDefinition()

    # Missing calibration result raises error
    context_missing = DecisionContext()
    with pytest.raises(DecisionExecutionError) as exc_info:
        policy.evaluate_metrics(context_missing, definition)
    assert "Calibration result is missing" in str(exc_info.value)

    # Success path (normalized binary Shannon entropy evaluation)
    cal_res = MockCalibrationResult(calibrated_confidence=0.5)
    context_success = DecisionContext(calibration_result=cal_res)
    metrics = policy.evaluate_metrics(context_success, definition)

    assert isinstance(metrics, DecisionMetrics)
    assert metrics.confidence == 0.5
    assert metrics.uncertainty == pytest.approx(1.0)  # Maximum entropy at p=0.5
    assert metrics.calibrated is True
    assert metrics.source == "entropy"


# --- Decision Metric Resolver Tests ---


def test_resolver_fallback_chain() -> None:
    registry = DecisionMetricPolicyRegistry(
        policies=(
            CalibratedMetricPolicy(),
            RawMetricPolicy(),
            EntropyMetricPolicy(),
        )
    )
    resolver = DecisionMetricResolver(
        registry=registry, default_confidence=0.4, default_uncertainty=0.6
    )

    definition = DecisionDefinition(confidence_policy="calibrated")

    # Case 1: All results present -> Calibration matches
    cal_res = MockCalibrationResult(calibrated_confidence=0.9)
    ver_res = MockVerificationResult(confidence=0.8)
    context_full = DecisionContext(
        calibration_result=cal_res, verification_result=ver_res
    )

    metrics = resolver.resolve_metrics(context_full, definition)
    assert metrics.confidence == 0.9
    assert metrics.source == "calibration"

    # Case 2: Calibration result missing -> Fallback to Raw Verification
    context_no_cal = DecisionContext(verification_result=ver_res)
    metrics = resolver.resolve_metrics(context_no_cal, definition)
    assert metrics.confidence == 0.8
    assert metrics.source == "verification"

    # Case 3: Both Calibration and Verification missing -> Fallback to default
    context_empty = DecisionContext()
    metrics = resolver.resolve_metrics(context_empty, definition)
    assert metrics.confidence == 0.4
    assert metrics.uncertainty == 0.6
    assert metrics.source == "default"


def test_resolver_entropy_fallback() -> None:
    registry = DecisionMetricPolicyRegistry(
        policies=(
            CalibratedMetricPolicy(),
            RawMetricPolicy(),
            EntropyMetricPolicy(),
        )
    )
    resolver = DecisionMetricResolver(registry=registry)

    # Configured to use entropy policy
    definition = DecisionDefinition(confidence_policy="entropy")

    cal_res = MockCalibrationResult(calibrated_confidence=0.5)
    context = DecisionContext(calibration_result=cal_res)

    metrics = resolver.resolve_metrics(context, definition)
    assert metrics.confidence == 0.5
    assert metrics.uncertainty == pytest.approx(1.0)
    assert metrics.source == "entropy"


# --- Decision Policy Engine Tests ---


def test_engine_metric_evaluation(
    dummy_verification_result: Any = None,
) -> None:
    # Build context with verification results
    ver_res = MockVerificationResult(confidence=0.95)
    context = DecisionContext(verification_result=ver_res)
    definition = DecisionDefinition(confidence_policy="raw")
    input_data = DecisionInput(context=context, definition=definition)

    engine = DecisionPolicyEngine()
    exec_ctx = engine.evaluate(input_data)

    assert exec_ctx.decision_metrics is not None
    assert exec_ctx.decision_metrics.confidence == 0.95
    assert exec_ctx.decision_metrics.source == "verification"


# --- Bootstrap Injection Tests ---


def test_bootstrap_builds_registries_correctly() -> None:
    config = Settings()
    registry = build_decision_registry(config)

    assert isinstance(registry, DecisionProfileRegistry)
    profile = registry.resolve("default_decision")

    # Engine must have injected metric resolver
    engine = profile.strategy.policy_engine
    assert isinstance(engine, DecisionPolicyEngine)
    assert isinstance(engine.metric_resolver, DecisionMetricResolver)
    assert isinstance(engine.metric_resolver.registry, DecisionMetricPolicyRegistry)


# --- End-to-End Integration Tests ---


def test_decision_modernized_metric_pipeline_integration() -> None:
    # Setup full context
    cal_res = MockCalibrationResult(calibrated_confidence=0.9)
    context = DecisionContext(calibration_result=cal_res)
    definition = DecisionDefinition(confidence_policy="calibrated")

    # Strategy and engine
    engine = DecisionPolicyEngine()
    strategy = PolicyDecisionStrategy(policy_engine=engine)

    result = strategy.decide(context, definition)

    assert isinstance(result, DecisionResult)
    assert result.final_confidence == 0.9
    assert result.final_uncertainty == pytest.approx(0.1)
