"""Unit tests for the M14.3 Evaluation Profiles."""

import pytest
from pydantic import ValidationError

from src.core.decision.decision_models import (
    DecisionAction,
    DecisionMetadata,
    DecisionResult,
)
from src.core.evaluation.evaluation_models import (
    EvaluationDefinition,
    EvaluationProfile,
    EvaluationProfileRegistry,
    RuleBasedEvaluationDefinition,
)
from src.core.evaluation.evaluator import Evaluator
from src.core.evaluation.implementations import RuleBasedEvaluator
from src.core.exceptions import (
    DuplicateEvaluationProfileError,
    EvaluationConfigurationError,
    EvaluationProfileNotFoundError,
)
from src.core.explainability.explainability_models import (
    ExplanationMetadata,
    ExplanationResult,
    ExplanationSection,
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


def build_mock_explanation_result() -> ExplanationResult:
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
    dr = DecisionResult(
        action=DecisionAction.ACCEPT,
        rationale="test",
        uncertainty_result=ur,
        metadata=DecisionMetadata(strategy_id="test_dec"),
    )
    section = ExplanationSection(
        identifier="decision_section", title="Decision", content="Test"
    )
    return ExplanationResult(
        sections=(section,),
        decision_result=dr,
        metadata=ExplanationMetadata(strategy_id="test_expl"),
    )


def test_profile_immutability() -> None:
    profile = EvaluationProfile(
        profile_id="test_id",
        definition=RuleBasedEvaluationDefinition(),
        engine=RuleBasedEvaluator(),
    )
    with pytest.raises(ValidationError):
        profile.profile_id = "new_id"


def test_profile_compatibility_validation() -> None:
    EvaluationProfile(
        profile_id="valid",
        definition=RuleBasedEvaluationDefinition(),
        engine=RuleBasedEvaluator(),
    )

    class MockDef(EvaluationDefinition):
        pass

    with pytest.raises(
        EvaluationConfigurationError,
        match="RuleBasedEvaluator requires RuleBasedEvaluationDefinition",
    ):
        EvaluationProfile(
            profile_id="invalid",
            definition=MockDef(),
            engine=RuleBasedEvaluator(),
        )


def test_registry_immutability() -> None:
    profile = EvaluationProfile(
        profile_id="test_id",
        definition=RuleBasedEvaluationDefinition(),
        engine=RuleBasedEvaluator(),
    )
    registry = EvaluationProfileRegistry(profiles=(profile,))

    with pytest.raises(ValidationError):
        registry.profiles = ()


def test_registry_empty_validation() -> None:
    with pytest.raises(
        ValidationError, match="Tuple should have at least 1 item after validation"
    ):
        EvaluationProfileRegistry(profiles=())


def test_duplicate_profile_detection() -> None:
    profile1 = EvaluationProfile(
        profile_id="duplicate",
        definition=RuleBasedEvaluationDefinition(),
        engine=RuleBasedEvaluator(),
    )
    profile2 = EvaluationProfile(
        profile_id="duplicate",
        definition=RuleBasedEvaluationDefinition(),
        engine=RuleBasedEvaluator(),
    )

    with pytest.raises(
        DuplicateEvaluationProfileError,
        match="Duplicate profile_id detected: duplicate",
    ):
        EvaluationProfileRegistry(profiles=(profile1, profile2))


def test_profile_resolution() -> None:
    profile = EvaluationProfile(
        profile_id="target",
        definition=RuleBasedEvaluationDefinition(),
        engine=RuleBasedEvaluator(),
    )
    registry = EvaluationProfileRegistry(profiles=(profile,))

    resolved = registry.resolve("target")
    assert resolved is profile


def test_unknown_profile_resolution() -> None:
    profile = EvaluationProfile(
        profile_id="target",
        definition=RuleBasedEvaluationDefinition(),
        engine=RuleBasedEvaluator(),
    )
    registry = EvaluationProfileRegistry(profiles=(profile,))

    with pytest.raises(
        EvaluationProfileNotFoundError, match="Evaluation profile not found: unknown"
    ):
        registry.resolve("unknown")


def test_execution_equivalence() -> None:
    defn = RuleBasedEvaluationDefinition()
    engine = RuleBasedEvaluator()
    profile = EvaluationProfile(
        profile_id="eval_profile",
        definition=defn,
        engine=engine,
    )
    registry = EvaluationProfileRegistry(profiles=(profile,))

    er = build_mock_explanation_result()
    orchestrator = Evaluator()

    direct_res = engine.evaluate(er, defn)

    resolved_profile = registry.resolve("eval_profile")
    profile_res = orchestrator.evaluate(
        er, resolved_profile.definition, resolved_profile.engine
    )

    assert direct_res.metrics == profile_res.metrics
    assert direct_res.metadata == profile_res.metadata

    assert profile_res.explanation_result is er
    assert profile_res.explanation_result.decision_result is er.decision_result
