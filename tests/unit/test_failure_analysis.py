"""Unit tests for modernized Verification Failure Analysis (M3.1) subsystem."""

import pytest
from pydantic import ValidationError

from src.core.bootstrap import build_failure_analysis_registry
from src.core.config import Settings
from src.core.exceptions import (
    DuplicateFailureAnalysisProfileError,
    FailureAnalysisProfileNotFoundError,
)
from src.core.failure.failure_models import (
    FailureAnalysisDefinition,
    FailureAnalysisProfile,
    FailureAnalysisProfileRegistry,
    FailureCategory,
    FailureClassification,
    FailureDiagnostic,
    FailureRootCause,
    FailureSeverity,
    FailureTrace,
)
from src.core.failure.implementations import DefaultFailureAnalyzer
from src.core.retrieval.retrieval_models import (
    EvidenceBundle,
    EvidencePassage,
    RetrievalMetadata,
)
from src.core.verification.verification_models import (
    VerificationResult,
    VerificationVerdict,
)


@pytest.fixture
def dummy_evidence_bundle() -> EvidenceBundle:
    p1 = EvidencePassage(document_id="d1", span_id="s1", text="some text", score=0.9)
    return EvidenceBundle(
        claim="Test claim",
        passages=(p1,),
        metadata=RetrievalMetadata(strategy_id="test", top_k=1),
    )


@pytest.fixture
def dummy_verification_result(
    dummy_evidence_bundle: EvidenceBundle,
) -> VerificationResult:
    return VerificationResult(
        verdict=VerificationVerdict.SUPPORTED,
        confidence=0.9,
        evidence_bundle=dummy_evidence_bundle,
    )


def test_failure_enums_and_models() -> None:
    # 1. FailureSeverity
    assert FailureSeverity.INFO == "INFO"
    assert FailureSeverity.CRITICAL == "CRITICAL"

    # 2. FailureCategory
    assert FailureCategory.RETRIEVAL == "RETRIEVAL"
    assert FailureCategory.VERIFICATION == "VERIFICATION"

    # 3. FailureRootCause
    assert FailureRootCause.MISSING_EVIDENCE == "MISSING_EVIDENCE"
    assert FailureRootCause.LOW_CONFIDENCE == "LOW_CONFIDENCE"


def test_failure_classification_and_diagnostic() -> None:
    classification = FailureClassification(
        category=FailureCategory.RETRIEVAL,
        severity=FailureSeverity.CRITICAL,
        affected_subsystem="retrieval",
    )
    assert classification.category == FailureCategory.RETRIEVAL
    assert classification.severity == FailureSeverity.CRITICAL

    with pytest.raises(ValidationError):
        setattr(classification, "severity", FailureSeverity.LOW)

    diagnostic = FailureDiagnostic(
        root_cause=FailureRootCause.MISSING_EVIDENCE,
        diagnostic_summary="No evidence passages found",
        affected_artifacts=("evidence_bundle",),
    )
    assert diagnostic.root_cause == FailureRootCause.MISSING_EVIDENCE


def test_failure_trace() -> None:
    trace = FailureTrace(
        analyzer_execution_order=("DefaultFailureAnalyzer",),
        diagnostic_sequence=("check_retrieval",),
        classification_path=("RETRIEVAL",),
        inspected_artifacts=("evidence_bundle",),
        execution_metadata={"key": "val"},
    )
    assert trace.analyzer_execution_order == ("DefaultFailureAnalyzer",)
    assert trace.execution_metadata == {"key": "val"}


def test_default_analyzer_retrieval_failure() -> None:
    analyzer = DefaultFailureAnalyzer()
    definition = FailureAnalysisDefinition()

    # Empty bundle -> Retrieval Failure
    empty_bundle = EvidenceBundle(
        claim="Claim",
        passages=(),
        metadata=RetrievalMetadata(strategy_id="test", top_k=0),
    )
    ver_res = VerificationResult(
        verdict=VerificationVerdict.INSUFFICIENT,
        confidence=1.0,
        evidence_bundle=empty_bundle,
    )

    res = analyzer.analyze("Claim", ver_res, definition)
    assert res.classification.category == FailureCategory.RETRIEVAL
    assert res.classification.severity == FailureSeverity.CRITICAL
    assert res.diagnostic.root_cause == FailureRootCause.MISSING_EVIDENCE


def test_default_analyzer_verification_failure(
    dummy_evidence_bundle: EvidenceBundle,
) -> None:
    analyzer = DefaultFailureAnalyzer()
    definition = FailureAnalysisDefinition()

    # Low confidence -> Verification Failure
    ver_res = VerificationResult(
        verdict=VerificationVerdict.SUPPORTED,
        confidence=0.3,
        evidence_bundle=dummy_evidence_bundle,
    )

    res = analyzer.analyze("Claim", ver_res, definition)
    assert res.classification.category == FailureCategory.VERIFICATION
    assert res.classification.severity == FailureSeverity.HIGH
    assert res.diagnostic.root_cause == FailureRootCause.LOW_CONFIDENCE


def test_default_analyzer_contradictory_evidence(
    dummy_evidence_bundle: EvidenceBundle,
) -> None:
    analyzer = DefaultFailureAnalyzer()
    definition = FailureAnalysisDefinition()

    # Contradictory passages -> Aggregation Failure
    ver_res = VerificationResult(
        verdict=VerificationVerdict.SUPPORTED,
        confidence=0.9,
        evidence_bundle=dummy_evidence_bundle,
        supporting_passages=("s1",),
        contradicting_passages=("s2",),
    )

    res = analyzer.analyze("Claim", ver_res, definition)
    assert res.classification.category == FailureCategory.AGGREGATION
    assert res.classification.severity == FailureSeverity.MEDIUM
    assert res.diagnostic.root_cause == FailureRootCause.CONTRADICTORY_EVIDENCE


def test_registry_resolution_and_validation() -> None:
    defn = FailureAnalysisDefinition()
    analyzer = DefaultFailureAnalyzer()
    profile = FailureAnalysisProfile(
        profile_id="p1",
        definition=defn,
        analyzer=analyzer,
    )

    registry = FailureAnalysisProfileRegistry(profiles=(profile,))
    assert registry.resolve("p1") is profile

    # Resolution failure
    with pytest.raises(FailureAnalysisProfileNotFoundError):
        registry.resolve("invalid")

    # Duplicate detection
    with pytest.raises(DuplicateFailureAnalysisProfileError):
        FailureAnalysisProfileRegistry(profiles=(profile, profile))


def test_bootstrap_building() -> None:
    settings = Settings()
    registry = build_failure_analysis_registry(settings)
    assert registry.resolve("default_failure_analysis") is not None


def test_failure_analysis_determinism(
    dummy_verification_result: VerificationResult,
) -> None:
    analyzer = DefaultFailureAnalyzer()
    definition = FailureAnalysisDefinition()

    res1 = analyzer.analyze("Claim", dummy_verification_result, definition)
    res2 = analyzer.analyze("Claim", dummy_verification_result, definition)

    # Identical output check
    assert res1.classification == res2.classification
    assert res1.diagnostic == res2.diagnostic
    assert res1.trace.analyzer_execution_order == res2.trace.analyzer_execution_order
