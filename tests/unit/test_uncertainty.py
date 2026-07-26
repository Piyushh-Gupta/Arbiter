"""Unit tests for the M11.1 Uncertainty Framework."""

import pytest
from pydantic import ValidationError

from src.core.exceptions import UncertaintyConfigurationError, UncertaintyExecutionError
from src.core.failure_analysis.failure_analysis_models import (
    FailureAnalysisResult,
    FailureMetadata,
    FailureSeverity,
)
from src.core.retrieval.retrieval_models import EvidenceBundle, RetrievalMetadata
from src.core.uncertainty.base import BaseUncertaintyEstimator
from src.core.uncertainty.estimator import UncertaintyEstimator
from src.core.uncertainty.uncertainty_models import (
    UncertaintyDefinition,
    UncertaintyFactor,
    UncertaintyLevel,
    UncertaintyMetadata,
    UncertaintyResult,
)
from src.core.verification.verification_models import (
    VerificationLabel,
    VerificationMetadata,
    VerificationResult,
)


def build_failure_analysis_result() -> FailureAnalysisResult:
    vr = VerificationResult(
        label=VerificationLabel.SUPPORTS,
        confidence=0.9,
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


class MockUncertaintyDefinition(UncertaintyDefinition):
    param: int


class MockUncertaintyEstimator:
    def __init__(self, fail_predict: bool = False, fail_compat: bool = False):
        self.fail_predict = fail_predict
        self.fail_compat = fail_compat
        self.called_with_claim: str | None = None
        self.called_with_fa_result: FailureAnalysisResult | None = None

    def validate_compatibility(self, definition: UncertaintyDefinition) -> None:
        if self.fail_compat:
            raise UncertaintyConfigurationError("Incompatible definition")

    def estimate(
        self,
        claim: str,
        failure_analysis_result: FailureAnalysisResult,
        definition: UncertaintyDefinition,
    ) -> UncertaintyResult:
        if self.fail_predict:
            raise UncertaintyExecutionError("Estimation failed")

        self.called_with_claim = claim
        self.called_with_fa_result = failure_analysis_result

        factor = UncertaintyFactor(code="TEST", description="test")

        return UncertaintyResult(
            level=UncertaintyLevel.LOW,
            score=0.1,
            factors=frozenset([factor]),
            failure_analysis_result=failure_analysis_result,
            metadata=UncertaintyMetadata(strategy_id="mock"),
        )


def test_protocol_compliance() -> None:
    assert isinstance(MockUncertaintyEstimator(), BaseUncertaintyEstimator)


def test_model_immutability() -> None:
    meta = UncertaintyMetadata(strategy_id="test")
    with pytest.raises(ValidationError):
        meta.strategy_id = "changed"

    factor = UncertaintyFactor(code="C", description="d")
    with pytest.raises(ValidationError):
        factor.code = "new"

    defn = MockUncertaintyDefinition(param=1)
    with pytest.raises(ValidationError):
        defn.param = 2

    fa_res = build_failure_analysis_result()
    res = UncertaintyResult(
        level=UncertaintyLevel.LOW,
        score=0.5,
        factors=frozenset([factor]),
        failure_analysis_result=fa_res,
        metadata=meta,
    )
    with pytest.raises(ValidationError):
        res.score = 0.6


def test_score_boundary_validation() -> None:
    fa_res = build_failure_analysis_result()
    meta = UncertaintyMetadata(strategy_id="test")

    with pytest.raises(ValidationError):
        UncertaintyResult(
            level=UncertaintyLevel.LOW,
            score=-0.1,
            factors=frozenset(),
            failure_analysis_result=fa_res,
            metadata=meta,
        )

    with pytest.raises(ValidationError):
        UncertaintyResult(
            level=UncertaintyLevel.LOW,
            score=1.1,
            factors=frozenset(),
            failure_analysis_result=fa_res,
            metadata=meta,
        )


def test_orchestrator_delegation_and_equivalence() -> None:
    strategy = MockUncertaintyEstimator()
    orchestrator = UncertaintyEstimator()
    defn = MockUncertaintyDefinition(param=1)
    fa_res = build_failure_analysis_result()

    res = orchestrator.estimate("Claim", fa_res, defn, strategy)

    # Verify exact return payload matches strategy creation logic
    assert res.level == UncertaintyLevel.LOW
    assert res.score == 0.1
    assert res.metadata.strategy_id == "mock"
    assert len(res.factors) == 1

    # Verify inputs reached strategy
    assert strategy.called_with_claim == "Claim"
    assert strategy.called_with_fa_result is fa_res

    # Verify identity preservation
    assert res.failure_analysis_result is fa_res
    assert res.failure_analysis_result.verification_result is fa_res.verification_result


def test_fail_fast_propagation() -> None:
    # Orchestrator should NOT wrap, it should just propagate
    strategy = MockUncertaintyEstimator(fail_predict=True)
    orchestrator = UncertaintyEstimator()
    defn = MockUncertaintyDefinition(param=1)
    fa_res = build_failure_analysis_result()

    with pytest.raises(UncertaintyExecutionError, match="Estimation failed"):
        orchestrator.estimate("Claim", fa_res, defn, strategy)
