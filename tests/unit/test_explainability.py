"""Unit tests for the M13.1 Explainability Framework."""

import pytest
from pydantic import ValidationError

from src.core.decision.decision_models import (
    DecisionAction,
    DecisionMetadata,
    DecisionResult,
)
from src.core.exceptions import ExplanationConfigurationError, ExplanationExecutionError
from src.core.explainability.explainability_models import (
    ExplanationDefinition,
    ExplanationMetadata,
    ExplanationResult,
    ExplanationSection,
)
from src.core.explainability.explainer import Explainer
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


class MockExplanationDefinition(ExplanationDefinition):
    pass


class MockExplainer:
    def validate_compatibility(self, definition: ExplanationDefinition) -> None:
        if not isinstance(definition, MockExplanationDefinition):
            raise ExplanationConfigurationError("Invalid definition type")

    def explain(
        self,
        claim: str,
        decision_result: DecisionResult,
        definition: ExplanationDefinition,
    ) -> ExplanationResult:
        if not isinstance(definition, MockExplanationDefinition):
            raise ExplanationConfigurationError("Invalid definition")

        if claim == "fail":
            raise ExplanationExecutionError("Simulated failure")

        section = ExplanationSection(
            identifier="test_section",
            title="Test Section",
            content="This is a test explanation.",
        )

        return ExplanationResult(
            sections=(section,),
            decision_result=decision_result,
            metadata=ExplanationMetadata(strategy_id="mock_explainer"),
        )


def test_immutable_models() -> None:
    section = ExplanationSection(
        identifier="test_id", title="Test Title", content="Test Content"
    )

    with pytest.raises(ValidationError):
        section.title = "New Title"

    res = ExplanationResult(
        sections=(section,),
        decision_result=build_mock_decision_result(),
        metadata=ExplanationMetadata(strategy_id="test"),
    )

    with pytest.raises(ValidationError):
        res.sections = ()


def test_sections_min_length() -> None:
    dr = build_mock_decision_result()
    meta = ExplanationMetadata(strategy_id="test")

    with pytest.raises(
        ValidationError, match="Tuple should have at least 1 item after validation"
    ):
        ExplanationResult(
            sections=(),
            decision_result=dr,
            metadata=meta,
        )


def test_orchestrator_delegation_and_equivalence() -> None:
    dr = build_mock_decision_result()
    definition = MockExplanationDefinition()
    strategy = MockExplainer()
    orchestrator = Explainer()

    # Direct execution
    direct_res = strategy.explain("claim", dr, definition)

    # Orchestrator execution
    orchestrator_res = orchestrator.explain("claim", dr, definition, strategy)

    # Equivalence
    assert direct_res.sections == orchestrator_res.sections
    assert direct_res.metadata == orchestrator_res.metadata

    # Identity preservation
    assert orchestrator_res.decision_result is dr
    assert orchestrator_res.decision_result.uncertainty_result is dr.uncertainty_result


def test_exception_propagation_and_fail_fast() -> None:
    dr = build_mock_decision_result()
    definition = MockExplanationDefinition()
    strategy = MockExplainer()
    orchestrator = Explainer()

    # The orchestrator should not catch or wrap exceptions raised by the strategy
    with pytest.raises(ExplanationExecutionError, match="Simulated failure"):
        orchestrator.explain("fail", dr, definition, strategy)
