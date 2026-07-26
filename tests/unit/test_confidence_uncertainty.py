"""Unit tests for the M11.2 Confidence-Based Uncertainty Estimator."""

import pytest
from pydantic import ValidationError

from src.core.exceptions import UncertaintyConfigurationError
from src.core.failure_analysis.failure_analysis_models import (
    FailureAnalysisResult,
    FailureMetadata,
    FailureSeverity,
)
from src.core.retrieval.retrieval_models import EvidenceBundle, RetrievalMetadata
from src.core.uncertainty.base import BaseUncertaintyEstimator
from src.core.uncertainty.estimator import UncertaintyEstimator
from src.core.uncertainty.implementations import ConfidenceUncertaintyEstimator
from src.core.uncertainty.uncertainty_models import (
    ConfidenceUncertaintyDefinition,
    UncertaintyDefinition,
    UncertaintyLevel,
)
from src.core.verification.verification_models import (
    VerificationLabel,
    VerificationMetadata,
    VerificationResult,
)


def build_failure_analysis_result(confidence: float | None) -> FailureAnalysisResult:
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
        failure_flags=frozenset(),
        severity=FailureSeverity.NONE,
        verification_result=vr,
        metadata=FailureMetadata(strategy_id="test_fa"),
    )


class MockOtherUncertaintyDefinition(UncertaintyDefinition):
    pass


def test_definition_immutability() -> None:
    defn = ConfidenceUncertaintyDefinition(
        none_threshold=0.1,
        low_threshold=0.3,
        medium_threshold=0.5,
        high_threshold=0.8,
    )
    with pytest.raises(ValidationError):
        defn.none_threshold = 0.2


def test_definition_strict_ordering() -> None:
    # Valid
    ConfidenceUncertaintyDefinition(
        none_threshold=0.1,
        low_threshold=0.3,
        medium_threshold=0.5,
        high_threshold=0.8,
    )

    # Invalid: non-decreasing but not strictly increasing
    with pytest.raises(ValidationError, match="strictly increasing"):
        ConfidenceUncertaintyDefinition(
            none_threshold=0.1,
            low_threshold=0.1,
            medium_threshold=0.5,
            high_threshold=0.8,
        )

    # Invalid: out of order
    with pytest.raises(ValidationError, match="strictly increasing"):
        ConfidenceUncertaintyDefinition(
            none_threshold=0.3,
            low_threshold=0.1,
            medium_threshold=0.5,
            high_threshold=0.8,
        )


def test_definition_bounds() -> None:
    # Invalid: below 0
    with pytest.raises(ValidationError):
        ConfidenceUncertaintyDefinition(
            none_threshold=-0.1,
            low_threshold=0.3,
            medium_threshold=0.5,
            high_threshold=0.8,
        )

    # Invalid: above 1
    with pytest.raises(ValidationError):
        ConfidenceUncertaintyDefinition(
            none_threshold=0.1,
            low_threshold=0.3,
            medium_threshold=0.5,
            high_threshold=1.1,
        )


def test_compatibility_validation() -> None:
    estimator = ConfidenceUncertaintyEstimator()

    # Valid
    defn = ConfidenceUncertaintyDefinition(
        none_threshold=0.1, low_threshold=0.3, medium_threshold=0.5, high_threshold=0.8
    )
    estimator.validate_compatibility(defn)

    # Invalid
    other_defn = MockOtherUncertaintyDefinition()
    with pytest.raises(
        UncertaintyConfigurationError, match="requires ConfidenceUncertaintyDefinition"
    ):
        estimator.validate_compatibility(other_defn)


def test_linear_inversion() -> None:
    estimator = ConfidenceUncertaintyEstimator()
    defn = ConfidenceUncertaintyDefinition(
        none_threshold=0.1, low_threshold=0.3, medium_threshold=0.5, high_threshold=0.8
    )
    fa_res = build_failure_analysis_result(confidence=0.8)  # score should be 0.2

    res = estimator.estimate("claim", fa_res, defn)

    assert res.score == pytest.approx(0.2)
    assert len(res.factors) == 0


def test_absent_confidence() -> None:
    estimator = ConfidenceUncertaintyEstimator()
    defn = ConfidenceUncertaintyDefinition(
        none_threshold=0.1, low_threshold=0.3, medium_threshold=0.5, high_threshold=0.8
    )
    fa_res = build_failure_analysis_result(confidence=None)

    res = estimator.estimate("claim", fa_res, defn)

    assert res.score == 1.0
    assert res.level == UncertaintyLevel.EXTREME
    assert len(res.factors) == 1
    factor = list(res.factors)[0]
    assert factor.code == "ABSENT_CONFIDENCE"


def test_boundary_classification() -> None:
    estimator = ConfidenceUncertaintyEstimator()
    defn = ConfidenceUncertaintyDefinition(
        none_threshold=0.1, low_threshold=0.3, medium_threshold=0.5, high_threshold=0.8
    )

    # EXACTLY none_threshold -> NONE
    fa_res = build_failure_analysis_result(confidence=0.9)  # score = 0.1
    res = estimator.estimate("claim", fa_res, defn)
    assert res.level == UncertaintyLevel.NONE

    # ABOVE none, <= low -> LOW
    fa_res = build_failure_analysis_result(confidence=0.8)  # score = 0.2
    res = estimator.estimate("claim", fa_res, defn)
    assert res.level == UncertaintyLevel.LOW

    fa_res = build_failure_analysis_result(confidence=0.7)  # score = 0.3
    res = estimator.estimate("claim", fa_res, defn)
    assert res.level == UncertaintyLevel.LOW

    # ABOVE low, <= medium -> MEDIUM
    fa_res = build_failure_analysis_result(confidence=0.6)  # score = 0.4
    res = estimator.estimate("claim", fa_res, defn)
    assert res.level == UncertaintyLevel.MEDIUM

    fa_res = build_failure_analysis_result(confidence=0.5)  # score = 0.5
    res = estimator.estimate("claim", fa_res, defn)
    assert res.level == UncertaintyLevel.MEDIUM

    # ABOVE medium, <= high -> HIGH
    fa_res = build_failure_analysis_result(confidence=0.3)  # score = 0.7
    res = estimator.estimate("claim", fa_res, defn)
    assert res.level == UncertaintyLevel.HIGH

    fa_res = build_failure_analysis_result(confidence=0.2)  # score = 0.8
    res = estimator.estimate("claim", fa_res, defn)
    assert res.level == UncertaintyLevel.HIGH

    # ABOVE high -> EXTREME
    fa_res = build_failure_analysis_result(confidence=0.1)  # score = 0.9
    res = estimator.estimate("claim", fa_res, defn)
    assert res.level == UncertaintyLevel.EXTREME


def test_orchestrator_delegation_and_equivalence() -> None:
    strategy = ConfidenceUncertaintyEstimator()
    orchestrator = UncertaintyEstimator()
    defn = ConfidenceUncertaintyDefinition(
        none_threshold=0.1, low_threshold=0.3, medium_threshold=0.5, high_threshold=0.8
    )
    fa_res = build_failure_analysis_result(confidence=0.6)  # score = 0.4 (MEDIUM)

    res_direct = strategy.estimate("claim", fa_res, defn)
    res_orch = orchestrator.estimate("claim", fa_res, defn, strategy)

    assert res_direct.score == res_orch.score
    assert res_direct.level == res_orch.level
    assert res_direct.metadata == res_orch.metadata

    # Verify FailureAnalysisResult identity preservation
    assert res_orch.failure_analysis_result is fa_res
    assert (
        res_orch.failure_analysis_result.verification_result
        is fa_res.verification_result
    )


def test_protocol_compliance() -> None:
    assert isinstance(ConfidenceUncertaintyEstimator(), BaseUncertaintyEstimator)
