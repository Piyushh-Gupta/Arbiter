"""Unit tests for the M10.5 Failure Analysis Profiles."""

import pytest
from pydantic import ValidationError

from src.core.exceptions import (
    DuplicateFailureAnalysisProfileError,
    FailureAnalysisConfigurationError,
    FailureAnalysisProfileNotFoundError,
)
from src.core.failure_analysis.analyzer import FailureAnalyzer
from src.core.failure_analysis.failure_analysis_models import (
    ContradictionAnalysisDefinition,
    FailureAnalysisProfile,
    FailureAnalysisProfileRegistry,
    RetrievalFailureAnalysisDefinition,
    VerificationFailureAnalysisDefinition,
)
from src.core.failure_analysis.implementations import (
    ContradictionAnalyzer,
    RetrievalFailureAnalyzer,
    VerificationFailureAnalyzer,
)
from src.core.retrieval.retrieval_models import EvidenceBundle, RetrievalMetadata
from src.core.verification.verification_models import (
    VerificationLabel,
    VerificationMetadata,
    VerificationResult,
)


def build_verification_result() -> VerificationResult:
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


def test_profile_immutability_and_validation() -> None:
    defn = RetrievalFailureAnalysisDefinition(min_passages=1)
    analyzer = RetrievalFailureAnalyzer()

    profile = FailureAnalysisProfile(
        profile_id="test_retrieval",
        definition=defn,
        analyzer=analyzer,
    )

    with pytest.raises(ValidationError):
        profile.profile_id = "changed"


def test_compatibility_validation_at_construction() -> None:
    # Mismatched definition and analyzer
    defn = ContradictionAnalysisDefinition(min_passage_label_confidence=0.8)
    analyzer = RetrievalFailureAnalyzer()

    with pytest.raises(FailureAnalysisConfigurationError):
        FailureAnalysisProfile(
            profile_id="invalid",
            definition=defn,
            analyzer=analyzer,
        )


def test_empty_registry_validation() -> None:
    with pytest.raises(ValidationError):
        FailureAnalysisProfileRegistry(profiles=())


def test_duplicate_profile_detection() -> None:
    defn = RetrievalFailureAnalysisDefinition(min_passages=1)
    analyzer = RetrievalFailureAnalyzer()

    profile1 = FailureAnalysisProfile(
        profile_id="duplicate",
        definition=defn,
        analyzer=analyzer,
    )
    profile2 = FailureAnalysisProfile(
        profile_id="duplicate",
        definition=defn,
        analyzer=analyzer,
    )

    with pytest.raises(
        DuplicateFailureAnalysisProfileError, match="Duplicate profile_id detected"
    ):
        FailureAnalysisProfileRegistry(profiles=(profile1, profile2))


def test_registry_immutability() -> None:
    defn = RetrievalFailureAnalysisDefinition(min_passages=1)
    analyzer = RetrievalFailureAnalyzer()

    profile = FailureAnalysisProfile(
        profile_id="test",
        definition=defn,
        analyzer=analyzer,
    )

    registry = FailureAnalysisProfileRegistry(profiles=(profile,))
    with pytest.raises(ValidationError):
        registry.profiles = ()


def test_successful_profile_resolution() -> None:
    defn = RetrievalFailureAnalysisDefinition(min_passages=1)
    analyzer = RetrievalFailureAnalyzer()

    profile = FailureAnalysisProfile(
        profile_id="test_retrieval",
        definition=defn,
        analyzer=analyzer,
    )

    registry = FailureAnalysisProfileRegistry(profiles=(profile,))
    resolved = registry.resolve("test_retrieval")

    assert resolved is profile


def test_unknown_profile_resolution() -> None:
    defn = RetrievalFailureAnalysisDefinition(min_passages=1)
    analyzer = RetrievalFailureAnalyzer()

    profile = FailureAnalysisProfile(
        profile_id="test",
        definition=defn,
        analyzer=analyzer,
    )

    registry = FailureAnalysisProfileRegistry(profiles=(profile,))

    with pytest.raises(FailureAnalysisProfileNotFoundError, match="not found: unknown"):
        registry.resolve("unknown")


def test_all_concrete_analyzers_registration() -> None:
    # 1. Retrieval
    retrieval_prof = FailureAnalysisProfile(
        profile_id="retrieval",
        definition=RetrievalFailureAnalysisDefinition(min_passages=1),
        analyzer=RetrievalFailureAnalyzer(),
    )

    # 2. Verification
    verification_prof = FailureAnalysisProfile(
        profile_id="verification",
        definition=VerificationFailureAnalysisDefinition(min_confidence_threshold=0.5),
        analyzer=VerificationFailureAnalyzer(),
    )

    # 3. Contradiction
    contradiction_prof = FailureAnalysisProfile(
        profile_id="contradiction",
        definition=ContradictionAnalysisDefinition(min_passage_label_confidence=0.8),
        analyzer=ContradictionAnalyzer(),
    )

    registry = FailureAnalysisProfileRegistry(
        profiles=(retrieval_prof, verification_prof, contradiction_prof)
    )

    assert registry.resolve("retrieval") is retrieval_prof
    assert registry.resolve("verification") is verification_prof
    assert registry.resolve("contradiction") is contradiction_prof


def test_execution_equivalence() -> None:
    defn = VerificationFailureAnalysisDefinition(min_confidence_threshold=0.99)
    analyzer = VerificationFailureAnalyzer()

    profile = FailureAnalysisProfile(
        profile_id="verification",
        definition=defn,
        analyzer=analyzer,
    )

    registry = FailureAnalysisProfileRegistry(profiles=(profile,))
    resolved = registry.resolve("verification")

    result = build_verification_result()

    # Direct analyzer call via profile
    res_profile = resolved.analyzer.analyze("Test claim", result, resolved.definition)

    # Orchestrator call
    orchestrator = FailureAnalyzer()
    res_orchestrator = orchestrator.analyze(
        "Test claim", result, resolved.definition, resolved.analyzer
    )

    assert res_profile.failure_flags == res_orchestrator.failure_flags
    assert res_profile.severity == res_orchestrator.severity
    assert res_profile.verification_result is res_orchestrator.verification_result
