"""Unit tests for the M10.2 Retrieval Failure Detector."""

import pytest
from pydantic import ValidationError

from src.core.exceptions import FailureAnalysisConfigurationError
from src.core.failure_analysis.analyzer import FailureAnalyzer
from src.core.failure_analysis.failure_analysis_models import (
    FailureAnalysisDefinition,
    FailureSeverity,
    RetrievalFailureAnalysisDefinition,
)
from src.core.failure_analysis.implementations import RetrievalFailureAnalyzer
from src.core.retrieval.retrieval_models import (
    EvidenceBundle,
    EvidencePassage,
    RetrievalMetadata,
)
from src.core.verification.verification_models import (
    VerificationLabel,
    VerificationMetadata,
    VerificationResult,
)


@pytest.fixture
def dummy_verification_result() -> VerificationResult:
    return VerificationResult(
        label=VerificationLabel.SUPPORTS,
        confidence=0.9,
        evidence_bundle=EvidenceBundle(
            claim="Test",
            passages=(),
            metadata=RetrievalMetadata(strategy_id="test", top_k=5),
        ),
        metadata=VerificationMetadata(strategy_id="mock"),
    )


def build_verification_result(
    passages: tuple[EvidencePassage, ...],
) -> VerificationResult:
    return VerificationResult(
        label=VerificationLabel.SUPPORTS,
        confidence=0.9,
        evidence_bundle=EvidenceBundle(
            claim="Test",
            passages=passages,
            metadata=RetrievalMetadata(strategy_id="test", top_k=5),
        ),
        metadata=VerificationMetadata(strategy_id="mock"),
    )


def test_definition_immutability_and_validation() -> None:
    # Valid
    defn = RetrievalFailureAnalysisDefinition(
        min_passages=3, min_score_threshold=0.5, min_unique_documents=2
    )

    with pytest.raises(ValidationError):
        defn.min_passages = 4

    # Invalid constraints
    with pytest.raises(ValidationError):
        RetrievalFailureAnalysisDefinition(min_passages=0)
    with pytest.raises(ValidationError):
        RetrievalFailureAnalysisDefinition(min_passages=3, min_score_threshold=-0.1)
    with pytest.raises(ValidationError):
        RetrievalFailureAnalysisDefinition(min_passages=3, min_score_threshold=1.1)
    with pytest.raises(ValidationError):
        RetrievalFailureAnalysisDefinition(min_passages=3, min_unique_documents=0)


def test_compatibility_validation() -> None:
    analyzer = RetrievalFailureAnalyzer()
    valid_defn = RetrievalFailureAnalysisDefinition(min_passages=1)
    analyzer.validate_compatibility(valid_defn)

    class OtherDef(FailureAnalysisDefinition):
        pass

    with pytest.raises(FailureAnalysisConfigurationError):
        analyzer.validate_compatibility(OtherDef())

    with pytest.raises(FailureAnalysisConfigurationError):
        # Also check at analyze time
        analyzer.analyze("Test", build_verification_result(()), OtherDef())


def test_empty_bundle_short_circuit() -> None:
    analyzer = RetrievalFailureAnalyzer()
    defn = RetrievalFailureAnalysisDefinition(
        min_passages=5, min_score_threshold=0.9, min_unique_documents=3
    )
    result = build_verification_result(())

    # Should short-circuit and emit ONLY EMPTY_BUNDLE
    fa_result = analyzer.analyze("Test", result, defn)

    assert fa_result.severity == FailureSeverity.CRITICAL
    assert len(fa_result.failure_flags) == 1
    flag = list(fa_result.failure_flags)[0]
    assert flag.code == "EMPTY_BUNDLE"

    # Identity preservation
    assert fa_result.verification_result is result


def test_insufficient_evidence() -> None:
    analyzer = RetrievalFailureAnalyzer()
    defn = RetrievalFailureAnalysisDefinition(min_passages=3)

    p1 = EvidencePassage(document_id="doc1", span_id="s1", text="T", score=0.9)
    p2 = EvidencePassage(document_id="doc2", span_id="s1", text="T", score=0.9)

    result = build_verification_result((p1, p2))
    fa_result = analyzer.analyze("Test", result, defn)

    assert fa_result.severity == FailureSeverity.HIGH
    assert len(fa_result.failure_flags) == 1
    assert list(fa_result.failure_flags)[0].code == "INSUFFICIENT_EVIDENCE"


def test_low_retrieval_scores() -> None:
    analyzer = RetrievalFailureAnalyzer()
    defn = RetrievalFailureAnalysisDefinition(min_passages=1, min_score_threshold=0.7)

    # All below threshold
    p1 = EvidencePassage(document_id="doc1", span_id="s1", text="T", score=0.6)
    p2 = EvidencePassage(document_id="doc2", span_id="s1", text="T", score=0.65)

    result = build_verification_result((p1, p2))
    fa_result = analyzer.analyze("Test", result, defn)

    assert fa_result.severity == FailureSeverity.MEDIUM
    assert len(fa_result.failure_flags) == 1
    assert list(fa_result.failure_flags)[0].code == "LOW_RETRIEVAL_SCORES"

    # One above threshold -> no flag
    p3 = EvidencePassage(document_id="doc3", span_id="s1", text="T", score=0.75)
    result_mixed = build_verification_result((p1, p3))
    fa_result_mixed = analyzer.analyze("Test", result_mixed, defn)
    assert len(fa_result_mixed.failure_flags) == 0
    assert fa_result_mixed.severity == FailureSeverity.NONE


def test_duplicate_evidence_structural_identity() -> None:
    analyzer = RetrievalFailureAnalyzer()
    defn = RetrievalFailureAnalysisDefinition(min_passages=1)

    # Duplicate by document_id and span_id
    p1 = EvidencePassage(document_id="doc1", span_id="s1", text="T1", score=0.9)
    p2 = EvidencePassage(document_id="doc1", span_id="s1", text="T2", score=0.8)

    result_dup = build_verification_result((p1, p2))
    fa_result_dup = analyzer.analyze("Test", result_dup, defn)
    assert "DUPLICATE_EVIDENCE" in [f.code for f in fa_result_dup.failure_flags]
    assert fa_result_dup.severity == FailureSeverity.LOW

    # Same text, different identity -> NO duplicate
    p3 = EvidencePassage(document_id="doc1", span_id="s1", text="SameText", score=0.9)
    p4 = EvidencePassage(document_id="doc2", span_id="s1", text="SameText", score=0.8)

    result_unique = build_verification_result((p3, p4))
    fa_result_unique = analyzer.analyze("Test", result_unique, defn)
    assert len(fa_result_unique.failure_flags) == 0


def test_low_evidence_diversity() -> None:
    analyzer = RetrievalFailureAnalyzer()
    defn = RetrievalFailureAnalysisDefinition(min_passages=1, min_unique_documents=2)

    # All from same document
    p1 = EvidencePassage(document_id="doc1", span_id="s1", text="T1", score=0.9)
    p2 = EvidencePassage(document_id="doc1", span_id="s2", text="T2", score=0.8)

    result = build_verification_result((p1, p2))
    fa_result = analyzer.analyze("Test", result, defn)
    assert "LOW_EVIDENCE_DIVERSITY" in [f.code for f in fa_result.failure_flags]
    assert fa_result.severity == FailureSeverity.LOW


def test_multiple_flags_and_severity_aggregation() -> None:
    analyzer = RetrievalFailureAnalyzer()
    # Require 5 passages, threshold 0.9, diversity 3
    defn = RetrievalFailureAnalysisDefinition(
        min_passages=5, min_score_threshold=0.9, min_unique_documents=3
    )

    # 3 passages (insufficient: HIGH)
    # scores: 0.5, 0.6, 0.7 (low score: MEDIUM)
    # doc1, doc1, doc1 (low diversity: LOW)
    # s1, s1, s2 (duplicate evidence doc1-s1: LOW)
    p1 = EvidencePassage(document_id="doc1", span_id="s1", text="T1", score=0.5)
    p2 = EvidencePassage(document_id="doc1", span_id="s1", text="T2", score=0.6)
    p3 = EvidencePassage(document_id="doc1", span_id="s2", text="T3", score=0.7)

    result = build_verification_result((p1, p2, p3))
    fa_result = analyzer.analyze("Test", result, defn)

    codes = {f.code for f in fa_result.failure_flags}
    assert codes == {
        "INSUFFICIENT_EVIDENCE",
        "LOW_RETRIEVAL_SCORES",
        "DUPLICATE_EVIDENCE",
        "LOW_EVIDENCE_DIVERSITY",
    }
    # Max severity should be HIGH
    assert fa_result.severity == FailureSeverity.HIGH


def test_orchestrator_equivalence() -> None:
    analyzer = RetrievalFailureAnalyzer()
    orchestrator = FailureAnalyzer()
    defn = RetrievalFailureAnalysisDefinition(min_passages=1)

    p1 = EvidencePassage(document_id="doc1", span_id="s1", text="T", score=0.9)
    result = build_verification_result((p1,))

    res1 = analyzer.analyze("Test", result, defn)
    res2 = orchestrator.analyze("Test", result, defn, analyzer)

    assert res1.failure_flags == res2.failure_flags
    assert res1.severity == res2.severity
    assert res1.verification_result is res2.verification_result
