"""Unit tests for the M12.2 Threshold-Based Decision Engine."""

import pytest
from pydantic import ValidationError

from src.core.decision.decision_models import (
    DecisionAction,
    DecisionDefinition,
    ThresholdDecisionDefinition,
)
from src.core.decision.engine import DecisionEngine
from src.core.decision.implementations import ThresholdDecisionEngine
from src.core.exceptions import DecisionConfigurationError
from src.core.failure_analysis.failure_analysis_models import (
    FailureAnalysisResult,
    FailureMetadata,
    FailureSeverity,
)
from src.core.retrieval.retrieval_models import EvidenceBundle, RetrievalMetadata
from src.core.uncertainty.uncertainty_models import (
    UncertaintyLevel,
    UncertaintyMetadata,
    UncertaintyResult,
)
from src.core.verification.verification_models import (
    VerificationLabel,
    VerificationMetadata,
    VerificationResult,
)


def build_mock_uncertainty_result(
    label: VerificationLabel, score: float
) -> UncertaintyResult:
    vr = VerificationResult(
        label=label,
        confidence=1.0 - score,
        evidence_bundle=EvidenceBundle(
            claim="Test",
            passages=(),
            metadata=RetrievalMetadata(strategy_id="test", top_k=5),
        ),
        metadata=VerificationMetadata(strategy_id="test_vr"),
    )
    fa = FailureAnalysisResult(
        failure_flags=frozenset(),
        severity=FailureSeverity.NONE,
        verification_result=vr,
        metadata=FailureMetadata(strategy_id="test_fa"),
    )
    return UncertaintyResult(
        level=UncertaintyLevel.LOW,
        score=score,
        factors=frozenset(),
        failure_analysis_result=fa,
        metadata=UncertaintyMetadata(strategy_id="test_unc"),
    )


def test_immutable_definition() -> None:
    defn = ThresholdDecisionDefinition(
        accept_max_uncertainty=0.2, reject_max_uncertainty=0.3
    )
    with pytest.raises(ValidationError):
        defn.accept_max_uncertainty = 0.5


def test_threshold_validation() -> None:
    # Valid
    ThresholdDecisionDefinition(accept_max_uncertainty=0.0, reject_max_uncertainty=1.0)

    # Invalid accept
    with pytest.raises(ValidationError):
        ThresholdDecisionDefinition(
            accept_max_uncertainty=1.1, reject_max_uncertainty=0.5
        )

    # Invalid reject
    with pytest.raises(ValidationError):
        ThresholdDecisionDefinition(
            accept_max_uncertainty=0.5, reject_max_uncertainty=-0.1
        )


def test_compatibility_validation() -> None:
    engine = ThresholdDecisionEngine()
    valid_def = ThresholdDecisionDefinition(
        accept_max_uncertainty=0.2, reject_max_uncertainty=0.3
    )
    engine.validate_compatibility(valid_def)

    class MockDef(DecisionDefinition):
        pass

    invalid_def = MockDef()
    with pytest.raises(
        DecisionConfigurationError,
        match="ThresholdDecisionEngine requires ThresholdDecisionDefinition",
    ):
        engine.validate_compatibility(invalid_def)


def test_routing_supports_within_threshold() -> None:
    engine = ThresholdDecisionEngine()
    defn = ThresholdDecisionDefinition(
        accept_max_uncertainty=0.3, reject_max_uncertainty=0.3
    )
    ur = build_mock_uncertainty_result(VerificationLabel.SUPPORTS, score=0.2)

    res = engine.decide("claim", ur, defn)
    assert res.action == DecisionAction.ACCEPT
    assert "Claim supported with uncertainty (0.2) <= threshold (0.3)" in res.rationale


def test_routing_supports_above_threshold() -> None:
    engine = ThresholdDecisionEngine()
    defn = ThresholdDecisionDefinition(
        accept_max_uncertainty=0.3, reject_max_uncertainty=0.3
    )
    ur = build_mock_uncertainty_result(VerificationLabel.SUPPORTS, score=0.4)

    res = engine.decide("claim", ur, defn)
    assert res.action == DecisionAction.ESCALATE
    assert (
        "exceeds acceptable thresholds or label (SUPPORTS) lacks deterministic routing"
        in res.rationale
    )


def test_routing_refutes_within_threshold() -> None:
    engine = ThresholdDecisionEngine()
    defn = ThresholdDecisionDefinition(
        accept_max_uncertainty=0.3, reject_max_uncertainty=0.4
    )
    ur = build_mock_uncertainty_result(VerificationLabel.REFUTES, score=0.35)

    res = engine.decide("claim", ur, defn)
    assert res.action == DecisionAction.REJECT
    assert "Claim refuted with uncertainty (0.35) <= threshold (0.4)" in res.rationale


def test_routing_refutes_above_threshold() -> None:
    engine = ThresholdDecisionEngine()
    defn = ThresholdDecisionDefinition(
        accept_max_uncertainty=0.3, reject_max_uncertainty=0.4
    )
    ur = build_mock_uncertainty_result(VerificationLabel.REFUTES, score=0.5)

    res = engine.decide("claim", ur, defn)
    assert res.action == DecisionAction.ESCALATE
    assert (
        "exceeds acceptable thresholds or label (REFUTES) lacks deterministic routing"
        in res.rationale
    )


def test_routing_not_enough_info_fallback() -> None:
    engine = ThresholdDecisionEngine()
    defn = ThresholdDecisionDefinition(
        accept_max_uncertainty=0.9, reject_max_uncertainty=0.9
    )
    # Even if score is very low (0.0), NOT_ENOUGH_INFO must route to ESCALATE
    ur = build_mock_uncertainty_result(VerificationLabel.NOT_ENOUGH_INFO, score=0.0)

    res = engine.decide("claim", ur, defn)
    assert res.action == DecisionAction.ESCALATE
    assert "or label (NOT_ENOUGH_INFO) lacks deterministic routing" in res.rationale


def test_inclusive_boundary_behavior() -> None:
    engine = ThresholdDecisionEngine()
    defn = ThresholdDecisionDefinition(
        accept_max_uncertainty=0.3, reject_max_uncertainty=0.3
    )

    # Exactly on boundary for SUPPORTS
    ur_supports = build_mock_uncertainty_result(VerificationLabel.SUPPORTS, score=0.3)
    res_supports = engine.decide("claim", ur_supports, defn)
    assert res_supports.action == DecisionAction.ACCEPT

    # Exactly on boundary for REFUTES
    ur_refutes = build_mock_uncertainty_result(VerificationLabel.REFUTES, score=0.3)
    res_refutes = engine.decide("claim", ur_refutes, defn)
    assert res_refutes.action == DecisionAction.REJECT


def test_execution_equivalence_and_identity_preservation() -> None:
    engine = ThresholdDecisionEngine()
    orchestrator = DecisionEngine()
    defn = ThresholdDecisionDefinition(
        accept_max_uncertainty=0.2, reject_max_uncertainty=0.2
    )
    ur = build_mock_uncertainty_result(VerificationLabel.SUPPORTS, score=0.1)

    direct_res = engine.decide("claim", ur, defn)
    orchestrator_res = orchestrator.decide("claim", ur, defn, engine)

    assert direct_res.action == orchestrator_res.action
    assert direct_res.rationale == orchestrator_res.rationale
    assert orchestrator_res.uncertainty_result is ur
