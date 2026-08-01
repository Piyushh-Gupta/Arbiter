"""Unit and integration tests for M2.7 Verification Explainability & Attribution Framework."""

import pytest

from src.core.bootstrap import (
    build_calibration_registry,
    build_explanation_registry,
    build_verification_registry,
)
from src.core.calibration.calibration_models import (
    CalibrationResult,
    CalibrationStrategyType,
    CalibrationTrace,
)
from src.core.config import Settings
from src.core.exceptions import (
    DuplicateExplanationProfileError,
    ExplanationConfigurationError,
)
from src.core.explainability.explainability_models import (
    ExplanationProfile,
    ExplanationProfileRegistry,
)
from src.core.explainability.explanation_models import VerificationExplanationDefinition
from src.core.explainability.implementations import (
    CompositeExplanationStrategy,
    ConfidenceExplanationStrategy,
    DecisionTraceStrategy,
    EvidenceAttributionStrategy,
)
from src.core.retrieval.retrieval_models import (
    EvidenceBundle,
    EvidencePassage,
    RetrievalMetadata,
)
from src.core.verification.verification_models import (
    AggregationTrace,
    ConflictAnalysis,
    VerificationResult,
    VerificationVerdict,
    VerifiedPassage,
)


@pytest.fixture
def dummy_verification_result() -> VerificationResult:
    p1 = EvidencePassage(document_id="d1", span_id="s1", text="text1", score=0.95)
    p2 = EvidencePassage(document_id="d2", span_id="s2", text="text2", score=0.80)
    vp1 = VerifiedPassage(
        passage=p1,
        label=VerificationVerdict.SUPPORTED,
        supports_score=0.9,
        refutes_score=0.05,
        not_enough_info_score=0.05,
    )
    vp2 = VerifiedPassage(
        passage=p2,
        label=VerificationVerdict.CONTRADICTED,
        supports_score=0.1,
        refutes_score=0.8,
        not_enough_info_score=0.1,
    )
    trace = AggregationTrace(
        aggregation_strategy="max_confidence",
        ordered_evaluation_sequence=("s1", "s2"),
        weighting_decisions={"s1": 1.0, "s2": 0.5},
        intermediate_scores={},
        final_decision_path="max confidence choice",
    )
    return VerificationResult(
        verdict=VerificationVerdict.SUPPORTED,
        confidence=0.9,
        supporting_passages=("s1",),
        contradicting_passages=("s2",),
        verified_passages=(vp1, vp2),
        aggregation_trace=trace,
        conflict_analysis=ConflictAnalysis(
            conflict_severity=0.5,
            resolution_rationale="resolved by weight",
        ),
    )


@pytest.fixture
def dummy_calibration_result() -> CalibrationResult:
    trace = CalibrationTrace(
        original_confidence=0.9,
        intermediate_values={},
        final_confidence=0.85,
        applied_strategy=CalibrationStrategyType.TEMPERATURE_SCALING,
        parameter_version="1.0",
    )
    return CalibrationResult(
        original_confidence=0.9,
        calibrated_confidence=0.85,
        uncertainty_estimate=0.15,
        calibration_trace=trace,
    )


@pytest.fixture
def dummy_evidence_bundle() -> EvidenceBundle:
    p1 = EvidencePassage(document_id="d1", span_id="s1", text="text1", score=0.95)
    p2 = EvidencePassage(document_id="d2", span_id="s2", text="text2", score=0.80)
    return EvidenceBundle(
        claim="Test claim",
        passages=(p1, p2),
        metadata=RetrievalMetadata(strategy_id="test", top_k=2),
    )


def test_evidence_attribution(
    dummy_verification_result: VerificationResult,
    dummy_calibration_result: CalibrationResult,
    dummy_evidence_bundle: EvidenceBundle,
) -> None:
    strategy = EvidenceAttributionStrategy()
    definition = VerificationExplanationDefinition(
        explanation_strategy="EVIDENCE_ATTRIBUTION"
    )

    result = strategy.explain(
        dummy_verification_result,
        dummy_calibration_result,
        dummy_evidence_bundle,
        dummy_verification_result.aggregation_trace,
        definition,
    )

    assert result.evidence_attribution is not None
    attr = result.evidence_attribution
    assert attr.supporting_passages == ("s1",)
    assert attr.contradicting_passages == ("s2",)
    assert attr.ignored_passages == ()
    assert attr.contribution_weights == {"s1": 1.0, "s2": 0.5}
    assert "evidence_attribution" in result.sections[0].identifier


def test_decision_trace(
    dummy_verification_result: VerificationResult,
    dummy_calibration_result: CalibrationResult,
    dummy_evidence_bundle: EvidenceBundle,
) -> None:
    strategy = DecisionTraceStrategy()
    definition = VerificationExplanationDefinition(
        explanation_strategy="DECISION_TRACE"
    )

    result = strategy.explain(
        dummy_verification_result,
        dummy_calibration_result,
        dummy_evidence_bundle,
        dummy_verification_result.aggregation_trace,
        definition,
    )

    assert result.decision_trace is not None
    trace = result.decision_trace
    assert trace.aggregation_strategy == "max_confidence"
    assert trace.calibration_strategy == "TEMPERATURE_SCALING"
    assert trace.confidence_evolution[0] == pytest.approx(0.85)
    assert trace.confidence_evolution[1] == pytest.approx(0.9)
    assert trace.confidence_evolution[2] == pytest.approx(0.85)
    assert trace.contradiction_resolution == "resolved by weight"
    assert "decision_trace" in result.sections[0].identifier


def test_confidence_explanation(
    dummy_verification_result: VerificationResult,
    dummy_calibration_result: CalibrationResult,
    dummy_evidence_bundle: EvidenceBundle,
) -> None:
    strategy = ConfidenceExplanationStrategy()
    definition = VerificationExplanationDefinition(
        explanation_strategy="CONFIDENCE_EXPLANATION"
    )

    result = strategy.explain(
        dummy_verification_result,
        dummy_calibration_result,
        dummy_evidence_bundle,
        dummy_verification_result.aggregation_trace,
        definition,
    )
    assert len(result.sections) == 1
    assert "confidence_explanation" in result.sections[0].identifier


def test_composite_explanation(
    dummy_verification_result: VerificationResult,
    dummy_calibration_result: CalibrationResult,
    dummy_evidence_bundle: EvidenceBundle,
) -> None:
    strategy = CompositeExplanationStrategy()
    definition = VerificationExplanationDefinition(explanation_strategy="COMPOSITE")

    result = strategy.explain(
        dummy_verification_result,
        dummy_calibration_result,
        dummy_evidence_bundle,
        dummy_verification_result.aggregation_trace,
        definition,
    )

    assert len(result.sections) == 3
    assert result.evidence_attribution is not None
    assert result.decision_trace is not None
    assert result.contribution_analysis is not None
    assert result.explanation_trace is not None

    contrib = result.contribution_analysis
    # retrieval: (0.95 + 0.80) / 2 = 0.875
    assert contrib.retrieval_contribution == 0.875
    assert pytest.approx(contrib.verification_contribution) == 0.85
    # aggregation: 1.0 + 0.5 = 1.5
    assert contrib.aggregation_contribution == 1.5
    # calibration: abs(0.9 - 0.85) = 0.05
    assert pytest.approx(contrib.calibration_contribution) == 0.05

    trace = result.explanation_trace
    assert trace.explanation_strategy == "COMPOSITE"
    assert trace.verification_profile == "nli-default"
    assert trace.aggregation_profile == "max_confidence"
    assert trace.calibration_profile == "TEMPERATURE_SCALING"
    assert trace.evidence_traversal == ("s1", "s2")
    assert trace.execution_order == ("Attribution", "Trace", "Confidence")


def test_registry_resolution_and_validation() -> None:
    engine = EvidenceAttributionStrategy()
    definition = VerificationExplanationDefinition(
        explanation_strategy="EVIDENCE_ATTRIBUTION"
    )
    profile = ExplanationProfile(
        profile_id="test_m27",
        definition=definition,
        engine=engine,
        verification_profile_id="default_verification",
        calibration_profile_id="identity",
    )

    registry = ExplanationProfileRegistry(profiles=(profile,))
    assert registry.resolve("test_m27") is profile

    # Test duplicate ID detection
    with pytest.raises(DuplicateExplanationProfileError):
        ExplanationProfileRegistry(profiles=(profile, profile))


def test_bootstrap_validations() -> None:
    settings = Settings()
    ver_reg = build_verification_registry(settings)
    cal_reg = build_calibration_registry(settings)

    registry = build_explanation_registry(settings, ver_reg, cal_reg)
    assert registry.resolve("composite_explanation") is not None
    assert registry.resolve("evidence_attribution") is not None

    # Test compatibility validation at startup (invalid referenced profile)
    class FakeRegistry:
        def resolve(self, profile_id: str) -> None:
            raise KeyError("Profile not found")

    with pytest.raises(ExplanationConfigurationError):
        build_explanation_registry(settings, FakeRegistry(), cal_reg)


def test_explainability_determinism(
    dummy_verification_result: VerificationResult,
    dummy_calibration_result: CalibrationResult,
    dummy_evidence_bundle: EvidenceBundle,
) -> None:
    strategy = CompositeExplanationStrategy()
    definition = VerificationExplanationDefinition(explanation_strategy="COMPOSITE")

    res1 = strategy.explain(
        dummy_verification_result,
        dummy_calibration_result,
        dummy_evidence_bundle,
        dummy_verification_result.aggregation_trace,
        definition,
    )
    res2 = strategy.explain(
        dummy_verification_result,
        dummy_calibration_result,
        dummy_evidence_bundle,
        dummy_verification_result.aggregation_trace,
        definition,
    )

    assert res1.sections == res2.sections
    assert res1.evidence_attribution == res2.evidence_attribution
    assert res1.decision_trace == res2.decision_trace
    assert res1.contribution_analysis == res2.contribution_analysis
    assert res1.explanation_trace == res2.explanation_trace
