"""Unit tests for M4.2 Decision Policies & Immutable Decision Models."""

import pytest
from pydantic import ValidationError

from src.core.bootstrap import build_decision_registry
from src.core.config import Settings
from src.core.decision import (
    BaseDecisionPolicyEngine,
    DecisionContext,
    DecisionDefinition,
    DecisionEngineMetadata,
    DecisionExecutionContext,
    DecisionExecutionMetadata,
    DecisionInput,
    DecisionPolicyEngine,
    DecisionPolicyGroup,
    DecisionPolicyResult,
    DecisionProfile,
    DecisionProfileRegistry,
    DecisionResult,
    DecisionRule,
    DecisionRuleEvaluation,
    DecisionRuntimeMetadata,
    PolicyDecisionStrategy,
)
from src.core.exceptions import (
    DecisionConfigurationError,
    DecisionProfileNotFoundError,
    DuplicateDecisionProfileError,
)
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
def dummy_context() -> DecisionContext:
    p1 = EvidencePassage(document_id="d1", span_id="s1", text="evidence", score=0.95)
    bundle = EvidenceBundle(
        claim="Test claim",
        passages=(p1,),
        metadata=RetrievalMetadata(strategy_id="test", top_k=1),
    )
    vr = VerificationResult(
        verdict=VerificationVerdict.SUPPORTED, confidence=0.9, evidence_bundle=bundle
    )
    return DecisionContext(verification_result=vr, metadata={"request_id": "req_m42"})


@pytest.fixture
def dummy_input(dummy_context: DecisionContext) -> DecisionInput:
    definition = DecisionDefinition()
    return DecisionInput(context=dummy_context, definition=definition)


# --- Model Immutability & Validation Tests ---


def test_m42_models_immutability(dummy_input: DecisionInput) -> None:
    group = DecisionPolicyGroup(group_id="g1", priority=10, enabled=True)
    with pytest.raises(ValidationError):
        setattr(group, "priority", 20)

    result = DecisionPolicyResult(
        group_id="g1", matched_rules=("r1",), selected_action="ACCEPT"
    )
    with pytest.raises(ValidationError):
        setattr(result, "selected_action", "REJECT")

    rule_eval = DecisionRuleEvaluation(rule_id="r1", matched=True, priority=100)
    with pytest.raises(ValidationError):
        setattr(rule_eval, "matched", False)

    runtime_meta = DecisionRuntimeMetadata(
        policy_engine="DecisionPolicyEngine",
        configuration_fingerprint="fp123",
        execution_timestamp="2026-08-01T00:00:00Z",
    )
    with pytest.raises(ValidationError):
        setattr(runtime_meta, "policy_engine", "OtherEngine")

    exec_meta = DecisionExecutionMetadata(
        request_id="req_1",
        execution_duration_ms=1.5,
        profile="default",
        decision_policy="calibrated",
    )
    with pytest.raises(ValidationError):
        setattr(exec_meta, "request_id", "req_2")

    engine_meta = DecisionEngineMetadata(
        evaluated_group_count=2, evaluated_rule_count=5
    )
    with pytest.raises(ValidationError):
        setattr(engine_meta, "evaluated_group_count", 3)

    exec_ctx = DecisionExecutionContext(
        ordered_policy_results=(result,),
        ordered_rule_evaluations=(rule_eval,),
        runtime_metadata=runtime_meta,
        execution_metadata=exec_meta,
        engine_metadata=engine_meta,
        selected_action="ACCEPT",
    )
    with pytest.raises(ValidationError):
        setattr(exec_ctx, "selected_action", "REJECT")


# --- Policy Group Ordering & Enablement Tests ---


def test_policy_group_ordering_and_enablement(dummy_input: DecisionInput) -> None:
    engine = DecisionPolicyEngine()

    r1 = DecisionRule(
        rule_id="r_low_group", priority=10, enabled=True, conditions={}, action="ACCEPT"
    )
    g_low = DecisionPolicyGroup(
        group_id="g_low", priority=10, enabled=True, ordered_rules=(r1,)
    )

    r2 = DecisionRule(
        rule_id="r_high_group",
        priority=100,
        enabled=True,
        conditions={},
        action="ESCALATE",
    )
    g_high = DecisionPolicyGroup(
        group_id="g_high", priority=100, enabled=True, ordered_rules=(r2,)
    )

    r3 = DecisionRule(
        rule_id="r_disabled", priority=200, enabled=True, conditions={}, action="REJECT"
    )
    g_disabled = DecisionPolicyGroup(
        group_id="g_disabled", priority=200, enabled=False, ordered_rules=(r3,)
    )

    ctx = engine.evaluate(dummy_input, policy_groups=(g_low, g_high, g_disabled))

    assert len(ctx.ordered_policy_results) == 2
    assert ctx.ordered_policy_results[0].group_id == "g_high"
    assert ctx.ordered_policy_results[1].group_id == "g_low"
    assert ctx.selected_action == "ESCALATE"


# --- Policy Engine & Execution Context Tests ---


def test_decision_policy_engine_interface(dummy_input: DecisionInput) -> None:
    engine = DecisionPolicyEngine()
    assert isinstance(engine, BaseDecisionPolicyEngine)

    engine.validate_compatibility(dummy_input.definition)

    exec_ctx = engine.evaluate(dummy_input)

    assert isinstance(exec_ctx, DecisionExecutionContext)
    assert exec_ctx.selected_action in ("ACCEPT", "REJECT", "ESCALATE", "ABSTAIN")
    assert exec_ctx.engine_metadata.evaluated_group_count > 0
    assert exec_ctx.engine_metadata.evaluated_rule_count > 0


# --- Strategy Orchestration Tests ---


def test_policy_decision_strategy_orchestration(dummy_context: DecisionContext) -> None:
    engine = DecisionPolicyEngine()
    strategy = PolicyDecisionStrategy(policy_engine=engine)

    result = strategy.decide(dummy_context)

    assert isinstance(result, DecisionResult)
    assert result.final_verdict in ("ACCEPT", "REJECT", "ESCALATE", "ABSTAIN")
    assert result.uncertainty_result is dummy_context.calibration_result
    assert result.metadata.strategy_id == "policy_decision_strategy"


def test_policy_strategy_invalid_definition_raises() -> None:
    strategy = PolicyDecisionStrategy()
    with pytest.raises(DecisionConfigurationError):
        strategy.validate_compatibility("invalid_definition")


# --- Registry Compatibility & Duplicate Tests ---


def test_registry_duplicate_ids_raises() -> None:
    def1 = DecisionDefinition()
    strat = PolicyDecisionStrategy()
    p1 = DecisionProfile(profile_id="dup_id", definition=def1, strategy=strat)
    p2 = DecisionProfile(profile_id="dup_id", definition=def1, strategy=strat)

    with pytest.raises(DuplicateDecisionProfileError):
        DecisionProfileRegistry(profiles=(p1, p2))


def test_registry_resolution() -> None:
    def1 = DecisionDefinition()
    strat = PolicyDecisionStrategy()
    p1 = DecisionProfile(profile_id="p1", definition=def1, strategy=strat)
    registry = DecisionProfileRegistry(profiles=(p1,))

    assert registry.resolve("p1").profile_id == "p1"
    with pytest.raises(DecisionProfileNotFoundError):
        registry.resolve("non_existent")


# --- Bootstrap Validation Tests ---


def test_bootstrap_build_decision_registry_m42() -> None:
    config = Settings()
    registry = build_decision_registry(config)

    assert isinstance(registry, DecisionProfileRegistry)
    profile = registry.resolve("default_decision")
    assert profile.profile_id == "default_decision"
    assert isinstance(profile.strategy, PolicyDecisionStrategy)


# --- End-to-End Execution Chain Test ---


def test_e2e_decision_execution_chain(dummy_context: DecisionContext) -> None:
    definition = DecisionDefinition()
    input_data = DecisionInput(context=dummy_context, definition=definition)

    engine = DecisionPolicyEngine()
    groups = PolicyDecisionStrategy.default_policy_groups()

    exec_context = engine.evaluate(input_data, groups)
    strategy = PolicyDecisionStrategy(policy_groups=groups, policy_engine=engine)

    res1 = strategy.decide(dummy_context, definition)
    res2 = strategy.decide(dummy_context, definition)

    assert exec_context.selected_action == res1.final_verdict
    assert res1.final_verdict == res2.final_verdict
    assert res1.final_confidence == res2.final_confidence
    assert res1.final_uncertainty == res2.final_uncertainty
