"""Unit tests for the M13.3 Explainability Profiles."""

import pytest
from pydantic import ValidationError

from src.core.decision.decision_models import (
    DecisionAction,
    DecisionMetadata,
    DecisionResult,
)
from src.core.exceptions import (
    DuplicateExplanationProfileError,
    ExplanationConfigurationError,
    ExplanationProfileNotFoundError,
)
from src.core.explainability.explainability_models import (
    ExplanationDefinition,
    ExplanationProfile,
    ExplanationProfileRegistry,
    RuleBasedExplanationDefinition,
)
from src.core.explainability.explainer import Explainer
from src.core.explainability.implementations import RuleBasedExplainer
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


def build_mock_decision_result() -> DecisionResult:
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
    ur = UncertaintyResult(
        level=UncertaintyLevel.LOW,
        score=0.1,
        factors=frozenset(),
        failure_analysis_result=fa,
        metadata=UncertaintyMetadata(strategy_id="test_unc"),
    )
    return DecisionResult(
        action=DecisionAction.ACCEPT,
        rationale="test",
        uncertainty_result=ur,
        metadata=DecisionMetadata(strategy_id="test_dec"),
    )


def test_profile_immutability() -> None:
    profile = ExplanationProfile(
        profile_id="test_id",
        definition=RuleBasedExplanationDefinition(),
        engine=RuleBasedExplainer(),
    )
    with pytest.raises(ValidationError):
        profile.profile_id = "new_id"


def test_profile_compatibility_validation() -> None:
    # Valid pair
    ExplanationProfile(
        profile_id="valid",
        definition=RuleBasedExplanationDefinition(),
        engine=RuleBasedExplainer(),
    )

    # Invalid pair
    class MockDef(ExplanationDefinition):
        pass

    with pytest.raises(
        ExplanationConfigurationError,
        match="RuleBasedExplainer requires RuleBasedExplanationDefinition",
    ):
        ExplanationProfile(
            profile_id="invalid",
            definition=MockDef(),
            engine=RuleBasedExplainer(),
        )


def test_registry_immutability() -> None:
    profile = ExplanationProfile(
        profile_id="test_id",
        definition=RuleBasedExplanationDefinition(),
        engine=RuleBasedExplainer(),
    )
    registry = ExplanationProfileRegistry(profiles=(profile,))

    with pytest.raises(ValidationError):
        registry.profiles = ()


def test_registry_empty_validation() -> None:
    with pytest.raises(
        ValidationError, match="Tuple should have at least 1 item after validation"
    ):
        ExplanationProfileRegistry(profiles=())


def test_duplicate_profile_detection() -> None:
    profile1 = ExplanationProfile(
        profile_id="duplicate",
        definition=RuleBasedExplanationDefinition(),
        engine=RuleBasedExplainer(),
    )
    profile2 = ExplanationProfile(
        profile_id="duplicate",
        definition=RuleBasedExplanationDefinition(),
        engine=RuleBasedExplainer(),
    )

    with pytest.raises(
        DuplicateExplanationProfileError,
        match="Duplicate profile_id detected: duplicate",
    ):
        ExplanationProfileRegistry(profiles=(profile1, profile2))


def test_profile_resolution() -> None:
    profile = ExplanationProfile(
        profile_id="target",
        definition=RuleBasedExplanationDefinition(),
        engine=RuleBasedExplainer(),
    )
    registry = ExplanationProfileRegistry(profiles=(profile,))

    resolved = registry.resolve("target")
    assert resolved is profile


def test_unknown_profile_resolution() -> None:
    profile = ExplanationProfile(
        profile_id="target",
        definition=RuleBasedExplanationDefinition(),
        engine=RuleBasedExplainer(),
    )
    registry = ExplanationProfileRegistry(profiles=(profile,))

    with pytest.raises(
        ExplanationProfileNotFoundError, match="Explanation profile not found: unknown"
    ):
        registry.resolve("unknown")


def test_execution_equivalence() -> None:
    # Setup profile and registry
    defn = RuleBasedExplanationDefinition()
    engine = RuleBasedExplainer()
    profile = ExplanationProfile(
        profile_id="explain_profile",
        definition=defn,
        engine=engine,
    )
    registry = ExplanationProfileRegistry(profiles=(profile,))

    # Execution context
    dr = build_mock_decision_result()
    orchestrator = Explainer()

    # Direct execution
    direct_res = engine.explain("claim", dr, defn)

    # Profile-driven execution
    resolved_profile = registry.resolve("explain_profile")
    profile_res = orchestrator.explain(
        "claim", dr, resolved_profile.definition, resolved_profile.engine
    )

    # Equivalence assertions
    assert direct_res.sections == profile_res.sections
    assert direct_res.metadata == profile_res.metadata

    # Object identity preservation
    assert profile_res.decision_result is dr
    assert profile_res.decision_result.uncertainty_result is dr.uncertainty_result
