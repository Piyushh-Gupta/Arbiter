"""Unit tests for the M11.4 Uncertainty Profiles."""

import pytest
from pydantic import ValidationError

from src.core.exceptions import (
    DuplicateUncertaintyProfileError,
    UncertaintyConfigurationError,
    UncertaintyProfileNotFoundError,
)
from src.core.failure_analysis.failure_analysis_models import (
    FailureAnalysisResult,
    FailureMetadata,
    FailureSeverity,
)
from src.core.retrieval.retrieval_models import EvidenceBundle, RetrievalMetadata
from src.core.uncertainty.estimator import UncertaintyEstimator
from src.core.uncertainty.implementations import (
    ConfidenceUncertaintyEstimator,
    FailureAwareUncertaintyEstimator,
)
from src.core.uncertainty.uncertainty_models import (
    ConfidenceUncertaintyDefinition,
    FailureAwareUncertaintyDefinition,
    UncertaintyProfile,
    UncertaintyProfileRegistry,
)
from src.core.verification.verification_models import (
    VerificationLabel,
    VerificationMetadata,
    VerificationResult,
)


def build_failure_analysis_result(confidence: float) -> FailureAnalysisResult:
    vr = VerificationResult(
        label=VerificationLabel.SUPPORTS,
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


def build_confidence_def() -> ConfidenceUncertaintyDefinition:
    return ConfidenceUncertaintyDefinition(
        none_threshold=0.1, low_threshold=0.3, medium_threshold=0.5, high_threshold=0.8
    )


def build_failure_aware_def() -> FailureAwareUncertaintyDefinition:
    return FailureAwareUncertaintyDefinition(
        none_threshold=0.1,
        low_threshold=0.3,
        medium_threshold=0.5,
        high_threshold=0.8,
        severity_penalties={FailureSeverity.HIGH: 0.5},
    )


def test_profile_immutability() -> None:
    profile = UncertaintyProfile(
        profile_id="test_id",
        definition=build_confidence_def(),
        estimator=ConfidenceUncertaintyEstimator(),
    )
    with pytest.raises(ValidationError):
        profile.profile_id = "new_id"


def test_profile_compatibility_validation() -> None:
    # Valid pair
    UncertaintyProfile(
        profile_id="valid",
        definition=build_confidence_def(),
        estimator=ConfidenceUncertaintyEstimator(),
    )

    # Invalid pair
    with pytest.raises(
        UncertaintyConfigurationError,
        match="requires FailureAwareUncertaintyDefinition",
    ):
        UncertaintyProfile(
            profile_id="invalid",
            definition=build_confidence_def(),
            estimator=FailureAwareUncertaintyEstimator(),
        )


def test_registry_immutability() -> None:
    profile = UncertaintyProfile(
        profile_id="test_id",
        definition=build_confidence_def(),
        estimator=ConfidenceUncertaintyEstimator(),
    )
    registry = UncertaintyProfileRegistry(profiles=(profile,))

    with pytest.raises(ValidationError):
        registry.profiles = ()


def test_registry_empty_validation() -> None:
    with pytest.raises(
        ValidationError, match="Tuple should have at least 1 item after validation"
    ):
        UncertaintyProfileRegistry(profiles=())


def test_duplicate_profile_detection() -> None:
    profile1 = UncertaintyProfile(
        profile_id="duplicate",
        definition=build_confidence_def(),
        estimator=ConfidenceUncertaintyEstimator(),
    )
    profile2 = UncertaintyProfile(
        profile_id="duplicate",
        definition=build_failure_aware_def(),
        estimator=FailureAwareUncertaintyEstimator(),
    )

    with pytest.raises(
        DuplicateUncertaintyProfileError,
        match="Duplicate profile_id detected: duplicate",
    ):
        UncertaintyProfileRegistry(profiles=(profile1, profile2))


def test_profile_resolution() -> None:
    profile = UncertaintyProfile(
        profile_id="target",
        definition=build_confidence_def(),
        estimator=ConfidenceUncertaintyEstimator(),
    )
    registry = UncertaintyProfileRegistry(profiles=(profile,))

    resolved = registry.resolve("target")
    assert resolved is profile


def test_unknown_profile_resolution() -> None:
    profile = UncertaintyProfile(
        profile_id="target",
        definition=build_confidence_def(),
        estimator=ConfidenceUncertaintyEstimator(),
    )
    registry = UncertaintyProfileRegistry(profiles=(profile,))

    with pytest.raises(
        UncertaintyProfileNotFoundError, match="Uncertainty profile not found: unknown"
    ):
        registry.resolve("unknown")


def test_execution_equivalence() -> None:
    # Setup profile and registry
    defn = build_failure_aware_def()
    estimator = FailureAwareUncertaintyEstimator()
    profile = UncertaintyProfile(
        profile_id="fa_profile",
        definition=defn,
        estimator=estimator,
    )
    registry = UncertaintyProfileRegistry(profiles=(profile,))

    # Execution context
    fa_res = build_failure_analysis_result(confidence=0.8)
    orchestrator = UncertaintyEstimator()

    # Direct execution
    direct_res = estimator.estimate("claim", fa_res, defn)

    # Profile-driven execution
    resolved_profile = registry.resolve("fa_profile")
    profile_res = orchestrator.estimate(
        "claim", fa_res, resolved_profile.definition, resolved_profile.estimator
    )

    # Equivalence assertions
    assert direct_res.score == profile_res.score
    assert direct_res.level == profile_res.level
    assert direct_res.metadata == profile_res.metadata

    # Object identity preservation
    assert profile_res.failure_analysis_result is fa_res
