"""Unit tests for the M12.1 Decision Framework."""

import pytest
from pydantic import ValidationError

from src.core.decision.decision_models import (
    DecisionAction,
    DecisionDefinition,
    DecisionMetadata,
    DecisionResult,
)
from src.core.decision.engine import DecisionEngine
from src.core.exceptions import DecisionConfigurationError, DecisionExecutionError
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


class MockDecisionDefinition(DecisionDefinition):
    threshold: float = 0.5


class MockDecisionEngine:
    def validate_compatibility(self, definition: DecisionDefinition) -> None:
        if not isinstance(definition, MockDecisionDefinition):
            raise DecisionConfigurationError("Invalid definition type")

    def decide(
        self,
        claim: str,
        uncertainty_result: UncertaintyResult,
        definition: DecisionDefinition,
    ) -> DecisionResult:
        if not isinstance(definition, MockDecisionDefinition):
            raise DecisionConfigurationError("Invalid definition")

        if claim == "fail":
            raise DecisionExecutionError("Simulated failure")

        action = (
            DecisionAction.ACCEPT
            if uncertainty_result.score < definition.threshold
            else DecisionAction.REJECT
        )
        return DecisionResult(
            action=action,
            rationale="Score below threshold",
            uncertainty_result=uncertainty_result,
            metadata=DecisionMetadata(strategy_id="mock_decision"),
        )


def test_immutable_models() -> None:
    ur = build_mock_uncertainty_result()
    res = DecisionResult(
        action=DecisionAction.ACCEPT,
        rationale="test",
        uncertainty_result=ur,
        metadata=DecisionMetadata(strategy_id="test_strat"),
    )

    with pytest.raises(ValidationError):
        res.action = DecisionAction.REJECT


def test_decision_action_enum() -> None:
    assert DecisionAction.ACCEPT.value == "ACCEPT"
    assert DecisionAction.REJECT.value == "REJECT"
    assert DecisionAction.ESCALATE.value == "ESCALATE"
    assert DecisionAction.ABSTAIN.value == "ABSTAIN"

    with pytest.raises(ValueError):
        DecisionAction("INVALID")


def test_orchestrator_delegation_and_equivalence() -> None:
    ur = build_mock_uncertainty_result()
    definition = MockDecisionDefinition(threshold=0.5)
    strategy = MockDecisionEngine()
    orchestrator = DecisionEngine()

    # Direct execution
    direct_res = strategy.decide("claim", ur, definition)

    # Orchestrator execution
    orchestrator_res = orchestrator.decide("claim", ur, definition, strategy)

    # Equivalence
    assert direct_res.action == orchestrator_res.action
    assert direct_res.rationale == orchestrator_res.rationale
    assert direct_res.metadata == orchestrator_res.metadata

    # Identity preservation (ensure we do not recreate or duplicate state)
    assert orchestrator_res.uncertainty_result is ur
    assert (
        orchestrator_res.uncertainty_result.failure_analysis_result
        is ur.failure_analysis_result
    )


def test_exception_propagation_and_fail_fast() -> None:
    ur = build_mock_uncertainty_result()
    definition = MockDecisionDefinition(threshold=0.5)
    strategy = MockDecisionEngine()
    orchestrator = DecisionEngine()

    # The orchestrator should not catch or wrap exceptions raised by the strategy
    with pytest.raises(DecisionExecutionError, match="Simulated failure"):
        orchestrator.decide("fail", ur, definition, strategy)
