"""Unit tests for the M14.2 Rule-Based Evaluator."""

import pytest
from pydantic import ValidationError

from src.core.decision.decision_models import (
    DecisionAction,
    DecisionMetadata,
    DecisionResult,
)
from src.core.evaluation.evaluation_models import (
    EvaluationDefinition,
    RuleBasedEvaluationDefinition,
)
from src.core.evaluation.evaluator import Evaluator
from src.core.evaluation.implementations import RuleBasedEvaluator
from src.core.exceptions import EvaluationConfigurationError
from src.core.explainability.explainability_models import (
    ExplanationMetadata,
    ExplanationResult,
    ExplanationSection,
)
from src.core.failure_analysis.failure_analysis_models import (
    FailureAnalysisResult,
    FailureFlag,
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


def build_pipeline_state(
    flags: frozenset[FailureFlag],
    severity: FailureSeverity,
    score: float,
    sections: tuple[ExplanationSection, ...],
) -> ExplanationResult:
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
        failure_flags=flags,
        severity=severity,
        verification_result=vr,
        metadata=FailureMetadata(strategy_id="test_fa"),
    )
    ur = UncertaintyResult(
        level=UncertaintyLevel.HIGH if score > 0.5 else UncertaintyLevel.LOW,
        score=score,
        factors=frozenset(),
        failure_analysis_result=fa,
        metadata=UncertaintyMetadata(strategy_id="test_unc"),
    )
    dr = DecisionResult(
        action=DecisionAction.ACCEPT,
        rationale="Test",
        uncertainty_result=ur,
        metadata=DecisionMetadata(strategy_id="test_dec"),
    )
    return ExplanationResult(
        sections=sections,
        decision_result=dr,
        metadata=ExplanationMetadata(strategy_id="test_expl"),
    )


def test_immutable_definition() -> None:
    defn = RuleBasedEvaluationDefinition()
    with pytest.raises(ValidationError):
        defn.new_attr = "test"  # type: ignore


def test_compatibility_validation() -> None:
    evaluator = RuleBasedEvaluator()
    valid_def = RuleBasedEvaluationDefinition()
    evaluator.validate_compatibility(valid_def)

    class MockDef(EvaluationDefinition):
        pass

    invalid_def = MockDef()
    with pytest.raises(
        EvaluationConfigurationError,
        match="RuleBasedEvaluator requires RuleBasedEvaluationDefinition",
    ):
        evaluator.validate_compatibility(invalid_def)


def test_deterministic_metric_ordering_and_completeness() -> None:
    sections = (
        ExplanationSection(identifier="decision_info", title="A", content="B"),
        ExplanationSection(identifier="uncertainty_info", title="A", content="B"),
        ExplanationSection(identifier="failure_info", title="A", content="B"),
        ExplanationSection(identifier="verification_info", title="A", content="B"),
    )
    er = build_pipeline_state(frozenset(), FailureSeverity.NONE, 0.0, sections)
    evaluator = RuleBasedEvaluator()
    defn = RuleBasedEvaluationDefinition()

    res = evaluator.evaluate(er, defn)

    assert len(res.metrics) == 3
    identifiers = [m.identifier for m in res.metrics]
    assert identifiers == [
        "structural_completeness",
        "uncertainty_confidence",
        "explanation_richness",
    ]


def test_structural_completeness_scoring() -> None:
    evaluator = RuleBasedEvaluator()
    defn = RuleBasedEvaluationDefinition()
    sections = (ExplanationSection(identifier="decision", title="A", content="B"),)

    # NONE severity -> 1.0
    er1 = build_pipeline_state(frozenset(), FailureSeverity.NONE, 0.0, sections)
    res1 = evaluator.evaluate(er1, defn)
    assert res1.metrics[0].score == 1.0

    # LOW severity -> 0.5
    er2 = build_pipeline_state(frozenset(), FailureSeverity.LOW, 0.0, sections)
    res2 = evaluator.evaluate(er2, defn)
    assert res2.metrics[0].score == 0.5

    # CRITICAL severity -> 0.0
    er3 = build_pipeline_state(frozenset(), FailureSeverity.CRITICAL, 0.0, sections)
    res3 = evaluator.evaluate(er3, defn)
    assert res3.metrics[0].score == 0.0


def test_uncertainty_confidence_scoring() -> None:
    evaluator = RuleBasedEvaluator()
    defn = RuleBasedEvaluationDefinition()
    sections = (ExplanationSection(identifier="decision", title="A", content="B"),)

    er1 = build_pipeline_state(frozenset(), FailureSeverity.NONE, 0.25, sections)
    res1 = evaluator.evaluate(er1, defn)
    assert res1.metrics[1].score == 0.75

    er2 = build_pipeline_state(frozenset(), FailureSeverity.NONE, 1.0, sections)
    res2 = evaluator.evaluate(er2, defn)
    assert res2.metrics[1].score == 0.0


def test_explanation_richness_scoring() -> None:
    evaluator = RuleBasedEvaluator()
    defn = RuleBasedEvaluationDefinition()

    # 4 domains covered
    sections_full = (
        ExplanationSection(identifier="my_decision_section", title="A", content="B"),
        ExplanationSection(identifier="uncertainty_factor_1", title="A", content="B"),
        ExplanationSection(identifier="failure_reason", title="A", content="B"),
        ExplanationSection(identifier="verification_label", title="A", content="B"),
    )
    er_full = build_pipeline_state(
        frozenset(), FailureSeverity.NONE, 0.0, sections_full
    )
    res_full = evaluator.evaluate(er_full, defn)
    assert res_full.metrics[2].score == 1.0

    # 2 domains covered
    sections_partial = (
        ExplanationSection(identifier="my_decision_section", title="A", content="B"),
        ExplanationSection(identifier="failure_reason", title="A", content="B"),
    )
    er_partial = build_pipeline_state(
        frozenset(), FailureSeverity.NONE, 0.0, sections_partial
    )
    res_partial = evaluator.evaluate(er_partial, defn)
    assert res_partial.metrics[2].score == 0.5

    # 0 domains covered
    sections_none = (
        ExplanationSection(identifier="unknown_xyz", title="A", content="B"),
    )
    er_none = build_pipeline_state(
        frozenset(), FailureSeverity.NONE, 0.0, sections_none
    )
    res_none = evaluator.evaluate(er_none, defn)
    assert res_none.metrics[2].score == 0.0


def test_execution_equivalence_and_identity_preservation() -> None:
    evaluator = RuleBasedEvaluator()
    orchestrator = Evaluator()
    defn = RuleBasedEvaluationDefinition()

    sections = (ExplanationSection(identifier="decision", title="A", content="B"),)
    er = build_pipeline_state(frozenset(), FailureSeverity.NONE, 0.0, sections)

    direct_res = evaluator.evaluate(er, defn)
    orchestrator_res = orchestrator.evaluate(er, defn, evaluator)

    assert direct_res.metrics == orchestrator_res.metrics
    assert direct_res.metadata == orchestrator_res.metadata
    assert orchestrator_res.explanation_result is er
    assert orchestrator_res.explanation_result.decision_result is er.decision_result
