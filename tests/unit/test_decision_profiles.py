"""Unit tests for the M12.3 Decision Profiles."""

import pytest
from pydantic import ValidationError

from src.core.decision.decision_models import (
    DecisionDefinition,
    DecisionProfile,
    DecisionProfileRegistry,
    ThresholdDecisionDefinition,
)
from src.core.decision.engine import DecisionEngine
from src.core.decision.implementations import ThresholdDecisionEngine
from src.core.exceptions import (
    DecisionConfigurationError,
    DecisionProfileNotFoundError,
    DuplicateDecisionProfileError,
)
from src.core.failure_analysis.failure_analysis_models import (
    FailureAnalysisResult,
    FailureMetadata,
    FailureSeverity,
)
from src.core.retrieval.retrieval_models import EvidenceBundle, RetrievalMetadata
from src.core.uncertainty.uncertainty_models import (
    UncertaintyLevel,
    UncertaintyMetadata,
    UncertaintyResult,
)
from src.core.verification.verification_models import (
    VerificationLabel,
    VerificationMetadata,
    VerificationResult,
)


def build_mock_uncertainty_result() -> UncertaintyResult:
    vr = VerificationResult(
        label=VerificationLabel.SUPPORTS,
        confidence=0.9,
        evidence_bundle=EvidenceBundle(
            claim="Test",
            passages=(),
            metadata=RetrievalMetadata(strategy_id="test", top_k=5),
        ),
        metadata=VerificationMetadata(strategy_id="test_vr"),
    )
    fa = FailureAnalysisResult(
        failure_flags=frozenset(),
        severity=FailureSeverity.NONE,
        verification_result=vr,
        metadata=FailureMetadata(strategy_id="test_fa"),
    )
    return UncertaintyResult(
        level=UncertaintyLevel.LOW,
        score=0.1,
        factors=frozenset(),
        failure_analysis_result=fa,
        metadata=UncertaintyMetadata(strategy_id="test_unc"),
    )


def test_profile_immutability() -> None:
    profile = DecisionProfile(
        profile_id="test_id",
        definition=ThresholdDecisionDefinition(
            accept_max_uncertainty=0.2, reject_max_uncertainty=0.3
        ),
        engine=ThresholdDecisionEngine(),
    )
    with pytest.raises(ValidationError):
        profile.profile_id = "new_id"


def test_profile_compatibility_validation() -> None:
    # Valid pair
    DecisionProfile(
        profile_id="valid",
        definition=ThresholdDecisionDefinition(
            accept_max_uncertainty=0.2, reject_max_uncertainty=0.3
        ),
        engine=ThresholdDecisionEngine(),
    )

    # Invalid pair
    class MockDef(DecisionDefinition):
        pass

    with pytest.raises(
        DecisionConfigurationError,
        match="ThresholdDecisionEngine requires ThresholdDecisionDefinition",
    ):
        DecisionProfile(
            profile_id="invalid",
            definition=MockDef(),
            engine=ThresholdDecisionEngine(),
        )


def test_registry_immutability() -> None:
    profile = DecisionProfile(
        profile_id="test_id",
        definition=ThresholdDecisionDefinition(
            accept_max_uncertainty=0.2, reject_max_uncertainty=0.3
        ),
        engine=ThresholdDecisionEngine(),
    )
    registry = DecisionProfileRegistry(profiles=(profile,))

    with pytest.raises(ValidationError):
        registry.profiles = ()


def test_registry_empty_validation() -> None:
    with pytest.raises(
        ValidationError, match="Tuple should have at least 1 item after validation"
    ):
        DecisionProfileRegistry(profiles=())


def test_duplicate_profile_detection() -> None:
    profile1 = DecisionProfile(
        profile_id="duplicate",
        definition=ThresholdDecisionDefinition(
            accept_max_uncertainty=0.2, reject_max_uncertainty=0.3
        ),
        engine=ThresholdDecisionEngine(),
    )
    profile2 = DecisionProfile(
        profile_id="duplicate",
        definition=ThresholdDecisionDefinition(
            accept_max_uncertainty=0.5, reject_max_uncertainty=0.5
        ),
        engine=ThresholdDecisionEngine(),
    )

    with pytest.raises(
        DuplicateDecisionProfileError, match="Duplicate profile_id detected: duplicate"
    ):
        DecisionProfileRegistry(profiles=(profile1, profile2))


def test_profile_resolution() -> None:
    profile = DecisionProfile(
        profile_id="target",
        definition=ThresholdDecisionDefinition(
            accept_max_uncertainty=0.2, reject_max_uncertainty=0.3
        ),
        engine=ThresholdDecisionEngine(),
    )
    registry = DecisionProfileRegistry(profiles=(profile,))

    resolved = registry.resolve("target")
    assert resolved is profile


def test_unknown_profile_resolution() -> None:
    profile = DecisionProfile(
        profile_id="target",
        definition=ThresholdDecisionDefinition(
            accept_max_uncertainty=0.2, reject_max_uncertainty=0.3
        ),
        engine=ThresholdDecisionEngine(),
    )
    registry = DecisionProfileRegistry(profiles=(profile,))

    with pytest.raises(
        DecisionProfileNotFoundError, match="Decision profile not found: unknown"
    ):
        registry.resolve("unknown")


def test_execution_equivalence() -> None:
    # Setup profile and registry
    defn = ThresholdDecisionDefinition(
        accept_max_uncertainty=0.2, reject_max_uncertainty=0.3
    )
    engine = ThresholdDecisionEngine()
    profile = DecisionProfile(
        profile_id="decision_profile",
        definition=defn,
        engine=engine,
    )
    registry = DecisionProfileRegistry(profiles=(profile,))

    # Execution context
    ur = build_mock_uncertainty_result()
    orchestrator = DecisionEngine()

    # Direct execution
    direct_res = engine.decide("claim", ur, defn)

    # Profile-driven execution
    resolved_profile = registry.resolve("decision_profile")
    profile_res = orchestrator.decide(
        "claim", ur, resolved_profile.definition, resolved_profile.engine
    )

    # Equivalence assertions
    assert direct_res.action == profile_res.action
    assert direct_res.rationale == profile_res.rationale
    assert direct_res.metadata == profile_res.metadata

    # Object identity preservation
    assert profile_res.uncertainty_result is ur
