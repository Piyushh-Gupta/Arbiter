"""Unit tests for the M14.1 Evaluation Framework."""

import pytest
from pydantic import ValidationError

from src.core.decision.decision_models import (
    DecisionAction,
    DecisionMetadata,
    DecisionResult,
)
from src.core.evaluation.evaluation_models import (
    EvaluationDefinition,
    EvaluationMetadata,
    EvaluationMetric,
    EvaluationResult,
)
from src.core.evaluation.evaluator import Evaluator
from src.core.exceptions import EvaluationConfigurationError, EvaluationExecutionError
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
            claim="Test claim",
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
        rationale="test rationale",
        uncertainty_result=ur,
        metadata=DecisionMetadata(strategy_id="test_dec"),
    )
    section = ExplanationSection(
        identifier="summary", title="Summary", content="Test summary"
    )
    return ExplanationResult(
        sections=(section,),
        decision_result=dr,
        metadata=ExplanationMetadata(strategy_id="test_expl"),
    )


class MockEvaluationDefinition(EvaluationDefinition):
    pass


class MockEvaluator:
    def validate_compatibility(self, definition: EvaluationDefinition) -> None:
        if not isinstance(definition, MockEvaluationDefinition):
            raise EvaluationConfigurationError("Invalid definition type")

    def evaluate(
        self,
        explanation_result: ExplanationResult,
        definition: EvaluationDefinition,
    ) -> EvaluationResult:
        if not isinstance(definition, MockEvaluationDefinition):
            raise EvaluationConfigurationError("Invalid definition")

        if explanation_result.decision_result.rationale == "fail":
            raise EvaluationExecutionError("Simulated failure")

        metric = EvaluationMetric(
            identifier="test_metric",
            title="Test Metric",
            score=1.0,
            details="Perfect score",
        )

        return EvaluationResult(
            metrics=(metric,),
            explanation_result=explanation_result,
            metadata=EvaluationMetadata(strategy_id="mock_evaluator"),
        )


def test_immutable_models() -> None:
    metric = EvaluationMetric(identifier="test_id", title="Test Title", score=0.5)

    with pytest.raises(ValidationError):
        metric.score = 0.8

    res = EvaluationResult(
        metrics=(metric,),
        explanation_result=build_mock_explanation_result(),
        metadata=EvaluationMetadata(strategy_id="test"),
    )

    with pytest.raises(ValidationError):
        res.metrics = ()


def test_metrics_min_length() -> None:
    er = build_mock_explanation_result()
    meta = EvaluationMetadata(strategy_id="test")

    with pytest.raises(
        ValidationError, match="Tuple should have at least 1 item after validation"
    ):
        EvaluationResult(
            metrics=(),
            explanation_result=er,
            metadata=meta,
        )


def test_orchestrator_delegation_and_equivalence() -> None:
    er = build_mock_explanation_result()
    definition = MockEvaluationDefinition()
    strategy = MockEvaluator()
    orchestrator = Evaluator()

    # Direct execution
    direct_res = strategy.evaluate(er, definition)

    # Orchestrator execution
    orchestrator_res = orchestrator.evaluate(er, definition, strategy)

    # Equivalence
    assert direct_res.metrics == orchestrator_res.metrics
    assert direct_res.metadata == orchestrator_res.metadata

    # Identity preservation
    assert orchestrator_res.explanation_result is er
    assert orchestrator_res.explanation_result.decision_result is er.decision_result


def test_exception_propagation_and_fail_fast() -> None:
    er = build_mock_explanation_result()
    # Modify the private field to simulate a failure condition for the mock
    object.__setattr__(er.decision_result, "rationale", "fail")

    definition = MockEvaluationDefinition()
    strategy = MockEvaluator()
    orchestrator = Evaluator()

    # The orchestrator should not catch or wrap exceptions raised by the strategy
    with pytest.raises(EvaluationExecutionError, match="Simulated failure"):
        orchestrator.evaluate(er, definition, strategy)
