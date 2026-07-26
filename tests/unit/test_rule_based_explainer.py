"""Unit tests for the M13.2 Rule-Based Explainer."""

import pytest
from pydantic import ValidationError

from src.core.decision.decision_models import (
    DecisionAction,
    DecisionMetadata,
    DecisionResult,
)
from src.core.exceptions import ExplanationConfigurationError
from src.core.explainability.explainability_models import (
    ExplanationDefinition,
    RuleBasedExplanationDefinition,
)
from src.core.explainability.explainer import Explainer
from src.core.explainability.implementations import RuleBasedExplainer
from src.core.failure_analysis.failure_analysis_models import (
    FailureAnalysisResult,
    FailureFlag,
    FailureMetadata,
    FailureSeverity,
)
from src.core.retrieval.retrieval_models import EvidenceBundle, RetrievalMetadata
from src.core.uncertainty.uncertainty_models import (
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


def build_pipeline_state(
    flags: frozenset[FailureFlag],
    factors: frozenset[UncertaintyFactor],
    confidence: float | None = 0.9,
) -> DecisionResult:
    vr = VerificationResult(
        label=VerificationLabel.SUPPORTS,
        confidence=confidence,
        evidence_bundle=EvidenceBundle(
            claim="Test",
            passages=(),
            metadata=RetrievalMetadata(strategy_id="test", top_k=5),
        ),
        metadata=VerificationMetadata(strategy_id="test_vr"),
    )
    fa = FailureAnalysisResult(
        failure_flags=flags,
        severity=FailureSeverity.HIGH if flags else FailureSeverity.NONE,
        verification_result=vr,
        metadata=FailureMetadata(strategy_id="test_fa"),
    )
    ur = UncertaintyResult(
        level=UncertaintyLevel.HIGH if factors else UncertaintyLevel.LOW,
        score=0.8 if factors else 0.1,
        factors=factors,
        failure_analysis_result=fa,
        metadata=UncertaintyMetadata(strategy_id="test_unc"),
    )
    return DecisionResult(
        action=DecisionAction.REJECT if factors or flags else DecisionAction.ACCEPT,
        rationale="Test rationale string.",
        uncertainty_result=ur,
        metadata=DecisionMetadata(strategy_id="test_dec"),
    )


def test_immutable_definition() -> None:
    defn = RuleBasedExplanationDefinition()
    with pytest.raises(ValidationError):
        defn.new_attr = "test"  # type: ignore


def test_compatibility_validation() -> None:
    explainer = RuleBasedExplainer()
    valid_def = RuleBasedExplanationDefinition()
    explainer.validate_compatibility(valid_def)

    class MockDef(ExplanationDefinition):
        pass

    invalid_def = MockDef()
    with pytest.raises(
        ExplanationConfigurationError,
        match="RuleBasedExplainer requires RuleBasedExplanationDefinition",
    ):
        explainer.validate_compatibility(invalid_def)


def test_clean_pipeline_state_produces_four_sections() -> None:
    dr = build_pipeline_state(frozenset(), frozenset())
    explainer = RuleBasedExplainer()
    defn = RuleBasedExplanationDefinition()

    res = explainer.explain("claim", dr, defn)

    assert len(res.sections) == 4
    identifiers = [s.identifier for s in res.sections]
    assert identifiers == [
        "decision_rationale",
        "uncertainty_analysis",
        "failure_analysis",
        "verification_result",
    ]

    # Check rationale verbatim
    assert "Test rationale string." in res.sections[0].content


def test_pipeline_with_both_optional_sections() -> None:
    dr = build_pipeline_state(
        frozenset([FailureFlag(code="TEST_FLAG", description="A test flag.")]),
        frozenset([UncertaintyFactor(code="ABSENT_CONFIDENCE", description="x")]),
    )
    explainer = RuleBasedExplainer()
    defn = RuleBasedExplanationDefinition()

    res = explainer.explain("claim", dr, defn)

    assert len(res.sections) == 6
    identifiers = [s.identifier for s in res.sections]
    assert identifiers == [
        "decision_rationale",
        "uncertainty_analysis",
        "uncertainty_factors",
        "failure_analysis",
        "failure_flags",
        "verification_result",
    ]


def test_deterministic_sorting() -> None:
    # Use flags and factors that could hash out of order
    flag1 = FailureFlag(code="ZETA", description="Z")
    flag2 = FailureFlag(code="ALPHA", description="A")
    flag3 = FailureFlag(code="OMEGA", description="O")

    factor1 = UncertaintyFactor(code="FAILURE_PENALTY_APPLIED", description="x")
    factor2 = UncertaintyFactor(code="ABSENT_CONFIDENCE", description="x")

    dr = build_pipeline_state(
        frozenset([flag1, flag2, flag3]), frozenset([factor1, factor2])
    )
    explainer = RuleBasedExplainer()
    defn = RuleBasedExplanationDefinition()

    res = explainer.explain("claim", dr, defn)

    factors_content = [
        s.content for s in res.sections if s.identifier == "uncertainty_factors"
    ][0]
    flags_content = [
        s.content for s in res.sections if s.identifier == "failure_flags"
    ][0]

    # Verify exact string order
    assert "ABSENT_CONFIDENCE, FAILURE_PENALTY_APPLIED" in factors_content
    assert "ALPHA, OMEGA, ZETA" in flags_content


def test_absent_confidence_rendering() -> None:
    dr = build_pipeline_state(frozenset(), frozenset(), confidence=None)
    explainer = RuleBasedExplainer()
    defn = RuleBasedExplanationDefinition()

    res = explainer.explain("claim", dr, defn)

    vr_content = [
        s.content for s in res.sections if s.identifier == "verification_result"
    ][0]
    assert "Confidence: N/A" in vr_content


def test_execution_equivalence_and_identity_preservation() -> None:
    explainer = RuleBasedExplainer()
    orchestrator = Explainer()
    defn = RuleBasedExplanationDefinition()
    dr = build_pipeline_state(frozenset(), frozenset())

    direct_res = explainer.explain("claim", dr, defn)
    orchestrator_res = orchestrator.explain("claim", dr, defn, explainer)

    assert direct_res.sections == orchestrator_res.sections
    assert orchestrator_res.decision_result is dr
    assert orchestrator_res.decision_result.uncertainty_result is dr.uncertainty_result
