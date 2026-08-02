"""Unit tests for M3.7 Failure Explainability & Reporting subsystem."""

import pytest
from pydantic import ValidationError

from src.core.bootstrap import build_failure_explainability_registry
from src.core.config import Settings
from src.core.exceptions import (
    DuplicateFailureAnalysisProfileError,
    FailureAnalysisConfigurationError,
    FailureAnalysisProfileNotFoundError,
)
from src.core.failure.explainability import (
    CompositeFailureExplanationStrategy,
    DecisionTraceExplanationStrategy,
    FailureDecisionTrace,
    FailureEvidenceExplanation,
    FailureExplanationDefinition,
    FailureExplanationProfile,
    FailureExplanationProfileRegistry,
    FailureExplanationResult,
    FailureExplanationTemplate,
    FailureReportRenderer,
    SummaryExplanationStrategy,
    compute_explanation_fingerprint,
)
from src.core.failure.failure_models import (
    FailureAnalysisResult,
    FailureCategory,
    FailureClassification,
    FailureCorrelation,
    FailureCorrelationResult,
    FailureDiagnostic,
    FailureRootCause,
    FailureSeverity,
    FailureTrace,
    RootCauseResult,
    SeverityEvaluationResult,
)
from src.core.failure_analysis.failure_analysis_models import FailureMetadata
from src.core.failure_analysis.failure_analysis_models import (
    FailureSeverity as LegacyFailureSeverity,
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
def dummy_evidence_bundle() -> EvidenceBundle:
    p1 = EvidencePassage(document_id="d1", span_id="s1", text="some text", score=0.9)
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
def dummy_analysis_result(
    dummy_verification_result: VerificationResult,
) -> FailureAnalysisResult:
    return FailureAnalysisResult(
        classification=FailureClassification(
            category=FailureCategory.RETRIEVAL,
            severity=FailureSeverity.HIGH,
            affected_subsystem="retrieval",
        ),
        diagnostic=FailureDiagnostic(
            root_cause=FailureRootCause.LOW_RETRIEVAL_RECALL,
            diagnostic_summary="Insufficient passage retrieval recall",
            affected_artifacts=("passage_index",),
            recovery_recommendation="Increase top-k candidates",
        ),
        trace=FailureTrace(
            analyzer_execution_order=("RetrievalFailureAnalyzer",),
            diagnostic_sequence=("empty_check", "threshold_check"),
            inspected_artifacts=("passage_index",),
        ),
        failure_flags=frozenset(),
        severity=LegacyFailureSeverity.HIGH,
        verification_result=dummy_verification_result,
        metadata=FailureMetadata(strategy_id="test"),
    )


@pytest.fixture
def dummy_correlation_result() -> FailureCorrelationResult:
    return FailureCorrelationResult(
        correlation_graph=(
            FailureCorrelation(
                correlation_id="c1",
                source_failure="RETRIEVAL",
                target_failure="VERIFICATION",
                correlation_confidence=0.9,
            ),
        ),
        root_failures=("RETRIEVAL",),
        dependency_edges={"RETRIEVAL": ("VERIFICATION",)},
        summary="Retrieval failure triggered downstream verification failure.",
    )


@pytest.fixture
def dummy_root_cause_result() -> RootCauseResult:
    return RootCauseResult(
        primary_root_cause="LOW_RETRIEVAL_RECALL",
        contributing_failures=("CONTRADICTORY_EVIDENCE",),
        dependency_path=("RETRIEVAL", "VERIFICATION"),
        attribution_confidence=0.85,
    )


@pytest.fixture
def dummy_severity_result() -> SeverityEvaluationResult:
    return SeverityEvaluationResult(
        overall_severity=FailureSeverity.HIGH,
        contributing_severities=(FailureSeverity.HIGH, FailureSeverity.MEDIUM),
        escalation_required=True,
        escalation_reason="Critical path retrieval degradation",
        applied_rule="rule_retrieval_high",
        policy_trace=("rule_retrieval_high",),
    )


# --- Model Tests ---


def test_models_immutability(
    dummy_analysis_result: FailureAnalysisResult,
) -> None:
    definition = FailureExplanationDefinition()
    with pytest.raises(ValidationError):
        setattr(definition, "strategy", "invalid")

    template = FailureExplanationTemplate(template_id="t1")
    with pytest.raises(ValidationError):
        setattr(template, "verbosity", "detailed")

    evidence = FailureEvidenceExplanation(supporting_diagnostics=("d1",))
    with pytest.raises(ValidationError):
        setattr(evidence, "supporting_diagnostics", ("d2",))

    trace = FailureDecisionTrace(reasoning_chain=("step 1",))
    with pytest.raises(ValidationError):
        setattr(trace, "reasoning_chain", ("step 2",))


def test_compute_explanation_fingerprint() -> None:
    def1 = FailureExplanationDefinition(strategy="summary", verbosity="standard")
    def2 = FailureExplanationDefinition(strategy="summary", verbosity="standard")
    def3 = FailureExplanationDefinition(strategy="summary", verbosity="detailed")

    fp1 = compute_explanation_fingerprint(def1)
    fp2 = compute_explanation_fingerprint(def2)
    fp3 = compute_explanation_fingerprint(def3)

    assert fp1 == fp2
    assert fp1 != fp3


# --- Strategy Tests ---


def test_summary_explanation_strategy(
    dummy_analysis_result: FailureAnalysisResult,
    dummy_correlation_result: FailureCorrelationResult,
    dummy_root_cause_result: RootCauseResult,
    dummy_severity_result: SeverityEvaluationResult,
) -> None:
    strategy = SummaryExplanationStrategy()
    definition = FailureExplanationDefinition()

    result = strategy.explain(
        analysis_result=dummy_analysis_result,
        correlation_result=dummy_correlation_result,
        root_cause_result=dummy_root_cause_result,
        severity_result=dummy_severity_result,
        definition=definition,
    )

    assert isinstance(result, FailureExplanationResult)
    assert "Failure [RETRIEVAL]" in result.summary
    assert "Severity Level: HIGH" in result.detailed_explanation
    assert "Escalation Required: True" in result.detailed_explanation
    assert "LOW_RETRIEVAL_RECALL" in result.detailed_explanation
    assert result.metadata.strategy_id == "summary_explanation_strategy"


def test_decision_trace_explanation_strategy(
    dummy_analysis_result: FailureAnalysisResult,
    dummy_correlation_result: FailureCorrelationResult,
    dummy_root_cause_result: RootCauseResult,
    dummy_severity_result: SeverityEvaluationResult,
) -> None:
    strategy = DecisionTraceExplanationStrategy()
    definition = FailureExplanationDefinition()

    result = strategy.explain(
        analysis_result=dummy_analysis_result,
        correlation_result=dummy_correlation_result,
        root_cause_result=dummy_root_cause_result,
        severity_result=dummy_severity_result,
        definition=definition,
    )

    assert isinstance(result, FailureExplanationResult)
    assert "Decision Trace Explanation for RETRIEVAL" in result.summary
    assert len(result.decision_trace.reasoning_chain) >= 4
    assert result.metadata.strategy_id == "decision_trace_explanation_strategy"


def test_composite_explanation_strategy(
    dummy_analysis_result: FailureAnalysisResult,
    dummy_correlation_result: FailureCorrelationResult,
    dummy_root_cause_result: RootCauseResult,
    dummy_severity_result: SeverityEvaluationResult,
) -> None:
    s1 = SummaryExplanationStrategy()
    s2 = DecisionTraceExplanationStrategy()
    composite = CompositeFailureExplanationStrategy(strategies=(s1, s2))

    definition = FailureExplanationDefinition()
    composite.validate_compatibility(definition)

    result = composite.explain(
        analysis_result=dummy_analysis_result,
        correlation_result=dummy_correlation_result,
        root_cause_result=dummy_root_cause_result,
        severity_result=dummy_severity_result,
        definition=definition,
    )

    assert isinstance(result, FailureExplanationResult)
    assert result.metadata.strategy_id == "composite_failure_explanation_strategy"
    assert len(result.evidence_explanation.supporting_diagnostics) > 0
    assert len(result.decision_trace.reasoning_chain) > 0


def test_composite_strategy_empty_raises() -> None:
    with pytest.raises(FailureAnalysisConfigurationError):
        CompositeFailureExplanationStrategy(strategies=())


# --- Renderer Tests ---


def test_failure_report_renderer(
    dummy_analysis_result: FailureAnalysisResult,
    dummy_correlation_result: FailureCorrelationResult,
    dummy_root_cause_result: RootCauseResult,
    dummy_severity_result: SeverityEvaluationResult,
) -> None:
    strategy = SummaryExplanationStrategy()
    result = strategy.explain(
        analysis_result=dummy_analysis_result,
        correlation_result=dummy_correlation_result,
        root_cause_result=dummy_root_cause_result,
        severity_result=dummy_severity_result,
    )

    renderer = FailureReportRenderer()

    # Markdown
    md_output = renderer.render_markdown(result)
    assert "# Failure Analysis Explanation Report" in md_output
    assert "## Summary" in md_output
    assert "Failure [RETRIEVAL]" in md_output

    # JSON
    json_output = renderer.render_json(result)
    assert '"summary":' in json_output
    assert '"strategy_id": "summary_explanation_strategy"' in json_output

    # Plain Text
    txt_output = renderer.render_plain_text(result)
    assert "FAILURE ANALYSIS EXPLANATION REPORT" in txt_output

    # Plain Text with template
    template = FailureExplanationTemplate(
        template_id="t1",
        summary_template="[SUM] {summary}",
        detail_template="[DET] {details}",
    )
    txt_tmpl_output = renderer.render_plain_text(result, template=template)
    assert "[SUM] Failure [RETRIEVAL]" in txt_tmpl_output


# --- Registry Tests ---


def test_registry_duplicate_profile_id_raises() -> None:
    def1 = FailureExplanationDefinition()
    strat1 = SummaryExplanationStrategy()
    p1 = FailureExplanationProfile(profile_id="p1", definition=def1, strategy=strat1)
    p2 = FailureExplanationProfile(profile_id="p1", definition=def1, strategy=strat1)

    with pytest.raises(DuplicateFailureAnalysisProfileError):
        FailureExplanationProfileRegistry(profiles=(p1, p2))


def test_registry_resolution_and_missing() -> None:
    def1 = FailureExplanationDefinition()
    strat1 = SummaryExplanationStrategy()
    p1 = FailureExplanationProfile(profile_id="p1", definition=def1, strategy=strat1)
    registry = FailureExplanationProfileRegistry(profiles=(p1,))

    resolved = registry.resolve("p1")
    assert resolved.profile_id == "p1"

    with pytest.raises(FailureAnalysisProfileNotFoundError):
        registry.resolve("missing_profile")


# --- Bootstrap Tests ---


def test_bootstrap_build_failure_explainability_registry() -> None:
    config = Settings()
    registry = build_failure_explainability_registry(config)

    assert isinstance(registry, FailureExplanationProfileRegistry)
    profile = registry.resolve("default_failure_explainability")
    assert profile.profile_id == "default_failure_explainability"
    assert isinstance(profile.strategy, CompositeFailureExplanationStrategy)


# --- Determinism & Integration Tests ---


def test_explainability_determinism(
    dummy_analysis_result: FailureAnalysisResult,
    dummy_correlation_result: FailureCorrelationResult,
    dummy_root_cause_result: RootCauseResult,
    dummy_severity_result: SeverityEvaluationResult,
) -> None:
    strategy = SummaryExplanationStrategy()
    definition = FailureExplanationDefinition()
    renderer = FailureReportRenderer()

    res1 = strategy.explain(
        dummy_analysis_result,
        dummy_correlation_result,
        dummy_root_cause_result,
        dummy_severity_result,
        definition,
    )
    res2 = strategy.explain(
        dummy_analysis_result,
        dummy_correlation_result,
        dummy_root_cause_result,
        dummy_severity_result,
        definition,
    )

    assert res1.summary == res2.summary
    assert res1.detailed_explanation == res2.detailed_explanation
    assert res1.evidence_explanation == res2.evidence_explanation
    assert res1.decision_trace == res2.decision_trace

    # Normalize generation_timestamp for rendered report determinism comparison
    res2_identical_ts = res2.model_copy(update={"metadata": res1.metadata})
    md1 = renderer.render_markdown(res1)
    md2 = renderer.render_markdown(res2_identical_ts)
    assert md1 == md2
