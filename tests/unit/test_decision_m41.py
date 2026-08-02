"""Unit tests for M4.1 Decision Engine Architecture Modernization."""

import pytest
from pydantic import ValidationError

from src.core.bootstrap import build_decision_registry
from src.core.config import Settings
from src.core.decision import (
    BaseDecisionStrategy,
    DecisionContext,
    DecisionDefinition,
    DecisionMetadata,
    DecisionPolicyEngine,
    DecisionProfile,
    DecisionProfileRegistry,
    DecisionResult,
    DecisionRule,
    DecisionTrace,
    PolicyDecisionStrategy,
    compute_decision_fingerprint,
)
from src.core.exceptions import (
    DecisionConfigurationError,
    DecisionProfileNotFoundError,
    DuplicateDecisionProfileError,
)
from src.core.failure.failure_models import FailureSeverity, SeverityEvaluationResult
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
    p1 = EvidencePassage(document_id="d1", span_id="s1", text="text", score=0.9)
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


@pytest.fixture
def dummy_severity_result() -> SeverityEvaluationResult:
    return SeverityEvaluationResult(
        overall_severity=FailureSeverity.CRITICAL,
        contributing_severities=(FailureSeverity.CRITICAL,),
        escalation_required=True,
        escalation_reason="Critical failure detected",
        applied_rule="rule_critical",
        policy_trace=("rule_critical",),
    )


@pytest.fixture
def dummy_context(
    dummy_evidence_bundle: EvidenceBundle,
    dummy_verification_result: VerificationResult,
    dummy_severity_result: SeverityEvaluationResult,
) -> DecisionContext:
    return DecisionContext(
        evidence_bundle=dummy_evidence_bundle,
        verification_result=dummy_verification_result,
        severity_result=dummy_severity_result,
        metadata={"request_id": "r123"},
    )


# --- Model Tests ---


def test_decision_models_immutability() -> None:
    definition = DecisionDefinition()
    with pytest.raises(ValidationError):
        setattr(definition, "decision_strategy", "custom")

    rule = DecisionRule(rule_id="r1", action="ACCEPT")
    with pytest.raises(ValidationError):
        setattr(rule, "priority", 50)

    trace = DecisionTrace(selected_rule="r1")
    with pytest.raises(ValidationError):
        setattr(trace, "selected_rule", "r2")

    meta = DecisionMetadata(
        strategy_id="s1",
        configuration_fingerprint="abc",
        generation_timestamp="2026-08-01",
    )
    with pytest.raises(ValidationError):
        setattr(meta, "strategy_id", "s2")

    result = DecisionResult(
        final_verdict="ACCEPT",
        final_confidence=0.9,
        final_uncertainty=0.1,
        decision_trace=trace,
        metadata=meta,
    )
    with pytest.raises(ValidationError):
        setattr(result, "final_verdict", "REJECT")


def test_compute_decision_fingerprint() -> None:
    def1 = DecisionDefinition(
        decision_strategy="policy", confidence_policy="calibrated"
    )
    def2 = DecisionDefinition(
        decision_strategy="policy", confidence_policy="calibrated"
    )
    def3 = DecisionDefinition(decision_strategy="policy", confidence_policy="raw")

    fp1 = compute_decision_fingerprint(def1)
    fp2 = compute_decision_fingerprint(def2)
    fp3 = compute_decision_fingerprint(def3)

    assert fp1 == fp2
    assert fp1 != fp3


# --- Decision Rules & Policy Engine Tests ---


def test_decision_policy_engine_precedence_and_evaluation(
    dummy_context: DecisionContext,
) -> None:
    engine = DecisionPolicyEngine()

    r1 = DecisionRule(
        rule_id="r_low_prio",
        priority=10,
        enabled=True,
        conditions={},
        action="ACCEPT",
    )
    r2 = DecisionRule(
        rule_id="r_high_prio",
        priority=90,
        enabled=True,
        conditions={"require_escalation": True},
        action="ESCALATE",
    )
    r_disabled = DecisionRule(
        rule_id="r_disabled",
        priority=100,
        enabled=False,
        conditions={},
        action="REJECT",
    )

    action, trace = engine.evaluate(dummy_context, [r1, r2, r_disabled])

    assert action == "ESCALATE"
    assert trace.selected_rule == "r_high_prio"
    assert "r_disabled" not in trace.evaluated_rules
    assert trace.evaluated_rules[0] == "r_high_prio"


# --- Strategy Tests ---


def test_policy_decision_strategy_execution(
    dummy_context: DecisionContext,
) -> None:
    strategy = PolicyDecisionStrategy()
    definition = DecisionDefinition()

    strategy.validate_compatibility(definition)

    result = strategy.decide(dummy_context, definition)

    assert isinstance(result, DecisionResult)
    assert result.final_verdict == "ESCALATE"
    assert result.escalation_required is True
    assert result.metadata.strategy_id == "policy_decision_strategy"


def test_policy_decision_strategy_accept_path(
    dummy_evidence_bundle: EvidenceBundle,
) -> None:
    vr = VerificationResult(
        verdict=VerificationVerdict.SUPPORTED,
        confidence=0.95,
        evidence_bundle=dummy_evidence_bundle,
    )
    context = DecisionContext(verification_result=vr)

    strategy = PolicyDecisionStrategy()
    result = strategy.decide(context)

    assert result.final_verdict == "ACCEPT"
    assert result.escalation_required is False


def test_strategy_invalid_definition_raises() -> None:
    strategy = PolicyDecisionStrategy()
    with pytest.raises(DecisionConfigurationError):
        strategy.validate_compatibility("invalid_def")


# --- Registry Tests ---


def test_decision_registry_duplicate_raises() -> None:
    def1 = DecisionDefinition()
    strat = PolicyDecisionStrategy()
    p1 = DecisionProfile(profile_id="p1", definition=def1, strategy=strat)
    p2 = DecisionProfile(profile_id="p1", definition=def1, strategy=strat)

    with pytest.raises(DuplicateDecisionProfileError):
        DecisionProfileRegistry(profiles=(p1, p2))


def test_decision_registry_resolution() -> None:
    def1 = DecisionDefinition()
    strat = PolicyDecisionStrategy()
    p1 = DecisionProfile(profile_id="p1", definition=def1, strategy=strat)
    registry = DecisionProfileRegistry(profiles=(p1,))

    assert registry.resolve("p1").profile_id == "p1"

    with pytest.raises(DecisionProfileNotFoundError):
        registry.resolve("missing_profile")


# --- Bootstrap Tests ---


def test_bootstrap_build_decision_registry() -> None:
    config = Settings()
    registry = build_decision_registry(config)

    assert isinstance(registry, DecisionProfileRegistry)
    profile = registry.resolve("default_decision")
    assert profile.profile_id == "default_decision"
    assert isinstance(profile.strategy, PolicyDecisionStrategy)


# --- Determinism & Integration Tests ---


def test_decision_determinism_and_integration(
    dummy_context: DecisionContext,
) -> None:
    strategy = PolicyDecisionStrategy()
    definition = DecisionDefinition()

    res1 = strategy.decide(dummy_context, definition)
    res2 = strategy.decide(dummy_context, definition)

    assert res1.final_verdict == res2.final_verdict
    assert res1.final_confidence == res2.final_confidence
    assert res1.final_uncertainty == res2.final_uncertainty
    assert res1.decision_trace == res2.decision_trace

    # Verify model compatibility with BaseDecisionStrategy protocol
    assert isinstance(strategy, BaseDecisionStrategy)
