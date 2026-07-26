"""Unit tests for the M11.3 Failure-Aware Uncertainty Estimator."""

import pytest
from pydantic import ValidationError

from src.core.exceptions import UncertaintyConfigurationError
from src.core.failure_analysis.failure_analysis_models import (
    FailureAnalysisResult,
    FailureFlag,
    FailureMetadata,
    FailureSeverity,
)
from src.core.retrieval.retrieval_models import EvidenceBundle, RetrievalMetadata
from src.core.uncertainty.base import BaseUncertaintyEstimator
from src.core.uncertainty.estimator import UncertaintyEstimator
from src.core.uncertainty.implementations import FailureAwareUncertaintyEstimator
from src.core.uncertainty.uncertainty_models import (
    FailureAwareUncertaintyDefinition,
    UncertaintyDefinition,
    UncertaintyLevel,
)
from src.core.verification.verification_models import (
    VerificationLabel,
    VerificationMetadata,
    VerificationResult,
)


def build_failure_analysis_result(
    confidence: float | None,
    severity: FailureSeverity = FailureSeverity.NONE,
    failure_flags: frozenset[FailureFlag] = frozenset(),
) -> FailureAnalysisResult:
    vr = VerificationResult(
        label=VerificationLabel.SUPPORTS
        if confidence is not None
        else VerificationLabel.NOT_ENOUGH_INFO,
        confidence=confidence,
        evidence_bundle=EvidenceBundle(
            claim="Test",
            passages=(),
            metadata=RetrievalMetadata(strategy_id="test", top_k=5),
        ),
        metadata=VerificationMetadata(strategy_id="mock"),
    )
    return FailureAnalysisResult(
        failure_flags=failure_flags,
        severity=severity,
        verification_result=vr,
        metadata=FailureMetadata(strategy_id="test_fa"),
    )


class MockOtherUncertaintyDefinition(UncertaintyDefinition):
    pass


def test_definition_penalty_validation() -> None:
    # Valid
    FailureAwareUncertaintyDefinition(
        none_threshold=0.1,
        low_threshold=0.3,
        medium_threshold=0.5,
        high_threshold=0.8,
        severity_penalties={FailureSeverity.HIGH: 0.5},
        flag_penalties={"TEST_FLAG": 0.8},
    )

    # Invalid severity penalty < 0
    with pytest.raises(ValidationError, match="must be in \\[0.0, 1.0\\]"):
        FailureAwareUncertaintyDefinition(
            none_threshold=0.1,
            low_threshold=0.3,
            medium_threshold=0.5,
            high_threshold=0.8,
            severity_penalties={FailureSeverity.HIGH: -0.1},
        )

    # Invalid severity penalty > 1
    with pytest.raises(ValidationError, match="must be in \\[0.0, 1.0\\]"):
        FailureAwareUncertaintyDefinition(
            none_threshold=0.1,
            low_threshold=0.3,
            medium_threshold=0.5,
            high_threshold=0.8,
            severity_penalties={FailureSeverity.HIGH: 1.1},
        )

    # Invalid flag penalty < 0
    with pytest.raises(ValidationError, match="must be in \\[0.0, 1.0\\]"):
        FailureAwareUncertaintyDefinition(
            none_threshold=0.1,
            low_threshold=0.3,
            medium_threshold=0.5,
            high_threshold=0.8,
            flag_penalties={"TEST": -0.1},
        )

    # Inherited threshold validation still applies
    with pytest.raises(ValidationError, match="strictly increasing"):
        FailureAwareUncertaintyDefinition(
            none_threshold=0.3,
            low_threshold=0.1,
            medium_threshold=0.5,
            high_threshold=0.8,
        )


def test_compatibility_validation() -> None:
    estimator = FailureAwareUncertaintyEstimator()

    defn = FailureAwareUncertaintyDefinition(
        none_threshold=0.1, low_threshold=0.3, medium_threshold=0.5, high_threshold=0.8
    )
    estimator.validate_compatibility(defn)

    other_defn = MockOtherUncertaintyDefinition()
    with pytest.raises(
        UncertaintyConfigurationError,
        match="requires FailureAwareUncertaintyDefinition",
    ):
        estimator.validate_compatibility(other_defn)


def test_baseline_equivalence() -> None:
    # When no penalties apply, it should be equivalent to M11.2 (1.0 - confidence)
    estimator = FailureAwareUncertaintyEstimator()
    defn = FailureAwareUncertaintyDefinition(
        none_threshold=0.1, low_threshold=0.3, medium_threshold=0.5, high_threshold=0.8
    )
    fa_res = build_failure_analysis_result(confidence=0.8)  # score = 0.2

    res = estimator.estimate("claim", fa_res, defn)

    assert res.score == pytest.approx(0.2)
    assert len(res.factors) == 0


def test_severity_penalty_application() -> None:
    estimator = FailureAwareUncertaintyEstimator()
    defn = FailureAwareUncertaintyDefinition(
        none_threshold=0.1,
        low_threshold=0.3,
        medium_threshold=0.5,
        high_threshold=0.8,
        severity_penalties={FailureSeverity.HIGH: 0.5},
    )

    # confidence = 0.8 -> baseline certainty = 0.8
    # max penalty = 0.5
    # adjusted certainty = 0.8 * (1.0 - 0.5) = 0.4
    # score = 1.0 - 0.4 = 0.6
    fa_res = build_failure_analysis_result(
        confidence=0.8, severity=FailureSeverity.HIGH
    )

    res = estimator.estimate("claim", fa_res, defn)

    assert res.score == pytest.approx(0.6)
    assert res.level == UncertaintyLevel.HIGH
    assert len(res.factors) == 1
    assert list(res.factors)[0].code == "FAILURE_PENALTY_APPLIED"


def test_failure_flag_penalty_application() -> None:
    estimator = FailureAwareUncertaintyEstimator()
    defn = FailureAwareUncertaintyDefinition(
        none_threshold=0.1,
        low_threshold=0.3,
        medium_threshold=0.5,
        high_threshold=0.8,
        flag_penalties={"SPECIFIC_FLAG": 0.2},
    )

    flag = FailureFlag(code="SPECIFIC_FLAG", description="desc")
    # confidence = 0.9 -> baseline certainty = 0.9
    # max penalty = 0.2
    # adjusted certainty = 0.9 * (1.0 - 0.2) = 0.72
    # score = 1.0 - 0.72 = 0.28
    fa_res = build_failure_analysis_result(
        confidence=0.9, failure_flags=frozenset([flag])
    )

    res = estimator.estimate("claim", fa_res, defn)

    assert res.score == pytest.approx(0.28)
    assert res.level == UncertaintyLevel.LOW
    assert len(res.factors) == 1
    assert list(res.factors)[0].code == "FAILURE_PENALTY_APPLIED"


def test_maximum_penalty_selection_and_no_double_counting() -> None:
    estimator = FailureAwareUncertaintyEstimator()
    defn = FailureAwareUncertaintyDefinition(
        none_threshold=0.1,
        low_threshold=0.3,
        medium_threshold=0.5,
        high_threshold=0.8,
        severity_penalties={FailureSeverity.MEDIUM: 0.2},
        flag_penalties={"FLAG_1": 0.3, "FLAG_2": 0.4},
    )

    flag1 = FailureFlag(code="FLAG_1", description="desc")
    flag2 = FailureFlag(code="FLAG_2", description="desc")

    # max penalty among (0.2, 0.3, 0.4) is 0.4
    # confidence = 0.8
    # adjusted certainty = 0.8 * (1.0 - 0.4) = 0.48
    # score = 1.0 - 0.48 = 0.52
    fa_res = build_failure_analysis_result(
        confidence=0.8,
        severity=FailureSeverity.MEDIUM,
        failure_flags=frozenset([flag1, flag2]),
    )

    res = estimator.estimate("claim", fa_res, defn)

    assert res.score == pytest.approx(0.52)
    assert res.level == UncertaintyLevel.HIGH
    assert len(res.factors) == 1


def test_missing_confidence() -> None:
    estimator = FailureAwareUncertaintyEstimator()
    defn = FailureAwareUncertaintyDefinition(
        none_threshold=0.1,
        low_threshold=0.3,
        medium_threshold=0.5,
        high_threshold=0.8,
        severity_penalties={FailureSeverity.CRITICAL: 1.0},  # Even with penalties
    )
    fa_res = build_failure_analysis_result(
        confidence=None, severity=FailureSeverity.CRITICAL
    )

    res = estimator.estimate("claim", fa_res, defn)

    assert res.score == 1.0
    assert res.level == UncertaintyLevel.EXTREME
    assert len(res.factors) == 1
    assert list(res.factors)[0].code == "ABSENT_CONFIDENCE"


def test_mathematical_bounds_preservation() -> None:
    estimator = FailureAwareUncertaintyEstimator()
    # Apply a 1.0 penalty
    defn = FailureAwareUncertaintyDefinition(
        none_threshold=0.1,
        low_threshold=0.3,
        medium_threshold=0.5,
        high_threshold=0.8,
        severity_penalties={FailureSeverity.CRITICAL: 1.0},
    )

    # confidence = 0.9. Penalty = 1.0 -> certainty = 0.0 -> score = 1.0
    fa_res = build_failure_analysis_result(
        confidence=0.9, severity=FailureSeverity.CRITICAL
    )
    res = estimator.estimate("claim", fa_res, defn)
    assert res.score == 1.0


def test_orchestrator_delegation_and_equivalence() -> None:
    strategy = FailureAwareUncertaintyEstimator()
    orchestrator = UncertaintyEstimator()
    defn = FailureAwareUncertaintyDefinition(
        none_threshold=0.1, low_threshold=0.3, medium_threshold=0.5, high_threshold=0.8
    )
    fa_res = build_failure_analysis_result(confidence=0.6)

    res_direct = strategy.estimate("claim", fa_res, defn)
    res_orch = orchestrator.estimate("claim", fa_res, defn, strategy)

    assert res_direct.score == res_orch.score
    assert res_direct.level == res_orch.level

    # Verify FailureAnalysisResult identity preservation
    assert res_orch.failure_analysis_result is fa_res
    assert (
        res_orch.failure_analysis_result.verification_result
        is fa_res.verification_result
    )


def test_protocol_compliance() -> None:
    assert isinstance(FailureAwareUncertaintyEstimator(), BaseUncertaintyEstimator)
