"""Unit tests for the M10.4 Contradiction Analyzer."""

import pytest
from pydantic import ValidationError

from src.core.exceptions import FailureAnalysisConfigurationError
from src.core.failure_analysis.analyzer import FailureAnalyzer
from src.core.failure_analysis.failure_analysis_models import (
    ContradictionAnalysisDefinition,
    FailureAnalysisDefinition,
    FailureSeverity,
)
from src.core.failure_analysis.implementations import ContradictionAnalyzer
from src.core.retrieval.retrieval_models import (
    EvidenceBundle,
    EvidencePassage,
    RetrievalMetadata,
)
from src.core.verification.verification_models import (
    VerificationLabel,
    VerificationMetadata,
    VerificationResult,
    VerifiedPassage,
)


def create_dummy_passage(id_val: str) -> EvidencePassage:
    return EvidencePassage(
        document_id=f"doc_{id_val}",
        span_id=f"span_{id_val}",
        text=f"Passage {id_val}",
        score=0.9,
        metadata={},
    )


def build_verification_result(
    verified_passages: tuple[VerifiedPassage, ...] | None,
) -> VerificationResult:
    return VerificationResult(
        label=VerificationLabel.SUPPORTS,
        confidence=0.9,
        evidence_bundle=EvidenceBundle(
            claim="Test",
            passages=(),
            metadata=RetrievalMetadata(strategy_id="test", top_k=5),
        ),
        verified_passages=verified_passages,
        metadata=VerificationMetadata(strategy_id="mock"),
    )


def test_definition_immutability_and_validation() -> None:
    defn = ContradictionAnalysisDefinition(min_passage_label_confidence=0.8)

    with pytest.raises(ValidationError):
        defn.min_passage_label_confidence = 0.9

    with pytest.raises(ValidationError):
        ContradictionAnalysisDefinition(min_passage_label_confidence=-0.1)

    with pytest.raises(ValidationError):
        ContradictionAnalysisDefinition(min_passage_label_confidence=1.1)

    with pytest.raises(ValidationError):
        ContradictionAnalysisDefinition.model_validate({})


def test_verified_passage_immutability() -> None:
    vp = VerifiedPassage(
        passage=create_dummy_passage("1"),
        label=VerificationLabel.SUPPORTS,
        supports_score=0.9,
        refutes_score=0.1,
        not_enough_info_score=0.0,
    )
    with pytest.raises(ValidationError):
        vp.supports_score = 0.95


def test_compatibility_validation() -> None:
    analyzer = ContradictionAnalyzer()
    valid_defn = ContradictionAnalysisDefinition(min_passage_label_confidence=0.5)
    analyzer.validate_compatibility(valid_defn)

    class OtherDef(FailureAnalysisDefinition):
        pass

    with pytest.raises(FailureAnalysisConfigurationError):
        analyzer.validate_compatibility(OtherDef())

    with pytest.raises(FailureAnalysisConfigurationError):
        analyzer.analyze("Test", build_verification_result(None), OtherDef())


def test_missing_verified_passages_noop() -> None:
    analyzer = ContradictionAnalyzer()
    defn = ContradictionAnalysisDefinition(min_passage_label_confidence=0.5)

    result = build_verification_result(None)
    fa_result = analyzer.analyze("Test", result, defn)

    assert len(fa_result.failure_flags) == 0
    assert fa_result.severity == FailureSeverity.NONE
    assert fa_result.verification_result is result


def test_contradiction_detection() -> None:
    analyzer = ContradictionAnalyzer()
    defn = ContradictionAnalysisDefinition(min_passage_label_confidence=0.8)

    vp_supports = VerifiedPassage(
        passage=create_dummy_passage("1"),
        label=VerificationLabel.SUPPORTS,
        supports_score=0.9,
        refutes_score=0.1,
        not_enough_info_score=0.0,
    )
    vp_refutes = VerifiedPassage(
        passage=create_dummy_passage("2"),
        label=VerificationLabel.REFUTES,
        supports_score=0.0,
        refutes_score=0.9,
        not_enough_info_score=0.1,
    )

    result = build_verification_result((vp_supports, vp_refutes))
    fa_result = analyzer.analyze("Test", result, defn)

    assert len(fa_result.failure_flags) == 1
    assert list(fa_result.failure_flags)[0].code == "CONTRADICTORY_EVIDENCE"
    assert fa_result.severity == FailureSeverity.MEDIUM


def test_no_contradiction_case() -> None:
    analyzer = ContradictionAnalyzer()
    defn = ContradictionAnalysisDefinition(min_passage_label_confidence=0.8)

    vp_supports1 = VerifiedPassage(
        passage=create_dummy_passage("1"),
        label=VerificationLabel.SUPPORTS,
        supports_score=0.9,
        refutes_score=0.1,
        not_enough_info_score=0.0,
    )
    vp_supports2 = VerifiedPassage(
        passage=create_dummy_passage("2"),
        label=VerificationLabel.SUPPORTS,
        supports_score=0.85,
        refutes_score=0.05,
        not_enough_info_score=0.1,
    )

    result = build_verification_result((vp_supports1, vp_supports2))
    fa_result = analyzer.analyze("Test", result, defn)

    assert len(fa_result.failure_flags) == 0
    assert fa_result.severity == FailureSeverity.NONE


def test_below_threshold_ignored() -> None:
    analyzer = ContradictionAnalyzer()
    defn = ContradictionAnalysisDefinition(min_passage_label_confidence=0.8)

    vp_supports = VerifiedPassage(
        passage=create_dummy_passage("1"),
        label=VerificationLabel.SUPPORTS,
        supports_score=0.9,
        refutes_score=0.1,
        not_enough_info_score=0.0,
    )
    # Refutes but below 0.8 threshold
    vp_refutes_weak = VerifiedPassage(
        passage=create_dummy_passage("2"),
        label=VerificationLabel.REFUTES,
        supports_score=0.1,
        refutes_score=0.6,
        not_enough_info_score=0.3,
    )

    result = build_verification_result((vp_supports, vp_refutes_weak))
    fa_result = analyzer.analyze("Test", result, defn)

    # The weak refutes passage should be ignored for contradiction flagging
    assert len(fa_result.failure_flags) == 0
    assert fa_result.severity == FailureSeverity.NONE


def test_orchestrator_equivalence() -> None:
    analyzer = ContradictionAnalyzer()
    orchestrator = FailureAnalyzer()
    defn = ContradictionAnalysisDefinition(min_passage_label_confidence=0.5)

    vp_supports = VerifiedPassage(
        passage=create_dummy_passage("1"),
        label=VerificationLabel.SUPPORTS,
        supports_score=0.9,
        refutes_score=0.1,
        not_enough_info_score=0.0,
    )
    vp_refutes = VerifiedPassage(
        passage=create_dummy_passage("2"),
        label=VerificationLabel.REFUTES,
        supports_score=0.0,
        refutes_score=0.9,
        not_enough_info_score=0.1,
    )

    result = build_verification_result((vp_supports, vp_refutes))

    res1 = analyzer.analyze("Test", result, defn)
    res2 = orchestrator.analyze("Test", result, defn, analyzer)

    assert res1.failure_flags == res2.failure_flags
    assert res1.severity == res2.severity
    assert res1.verification_result is res2.verification_result
