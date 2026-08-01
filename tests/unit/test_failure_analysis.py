"""Unit tests for M3.3 Diagnostic Engines."""

from typing import Any

import pytest
from pydantic import ValidationError

from src.core.bootstrap import build_failure_analysis_registry
from src.core.calibration.calibration_models import (
    CalibrationResult,
    CalibrationStrategyType,
    CalibrationTrace,
)
from src.core.config import Settings
from src.core.failure.failure_models import (
    AnalyzerExecutionResult,
    DiagnosticEvidence,
    FailureAnalysisDefinition,
    FailureAnalysisInput,
    FailureArtifactReference,
    FailureCategory,
    FailureClassification,
    FailureDiagnosticContext,
    FailureExecutionMetadata,
    FailureRootCause,
    FailureRuntimeMetadata,
    FailureSeverity,
)
from src.core.failure.implementations import (
    CalibrationFailureAnalyzer,
    CompositeFailureAnalyzer,
    DefaultFailureAggregationStrategy,
    DefaultFailureAnalyzer,
    InfrastructureFailureAnalyzer,
    RetrievalFailureAnalyzer,
    VerificationFailureAnalyzer,
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


def test_models_immutability(dummy_evidence_bundle: EvidenceBundle) -> None:
    ref = FailureArtifactReference(
        artifact_id="a1",
        artifact_type="bundle",
        subsystem="retrieval",
    )
    with pytest.raises(ValidationError):
        setattr(ref, "subsystem", "verification")

    runtime = FailureRuntimeMetadata(
        analyzer_id="a_id",
        analyzer_version="1.0",
        execution_environment="production",
        execution_device="cpu",
        framework="python",
        execution_timestamp="2026-08-01",
    )
    with pytest.raises(ValidationError):
        setattr(runtime, "execution_device", "gpu")

    exec_meta = FailureExecutionMetadata(
        request_id="req1",
        execution_duration=12.5,
        analyzer_profile="default",
        configuration_fingerprint="abc",
    )
    with pytest.raises(ValidationError):
        setattr(exec_meta, "execution_duration", 15.0)

    context = FailureDiagnosticContext(
        ordered_analyzer_outputs=(),
        aggregated_metadata={},
    )
    with pytest.raises(ValidationError):
        setattr(context, "ordered_analyzer_outputs", ("out",))


def test_retrieval_failure_analyzer() -> None:
    analyzer = RetrievalFailureAnalyzer()
    defn = FailureAnalysisDefinition()

    # Case 1: Empty evidence bundle -> should trigger Retrieval Failure
    empty_result = VerificationResult(
        verdict=VerificationVerdict.CONTRADICTED,
        confidence=0.9,
        evidence_bundle=EvidenceBundle(
            claim="Empty",
            passages=(),
            metadata=RetrievalMetadata(strategy_id="test", top_k=0),
        ),
    )
    inp = FailureAnalysisInput(
        claim="Empty claim",
        pipeline_artifacts={"verification_result": empty_result},
        definition=defn,
    )
    res = analyzer.analyze(inp)
    assert res.classification.category == FailureCategory.RETRIEVAL
    assert res.classification.severity == FailureSeverity.CRITICAL
    assert res.diagnostic.root_cause == FailureRootCause.MISSING_EVIDENCE


def test_verification_failure_analyzer(dummy_evidence_bundle: EvidenceBundle) -> None:
    analyzer = VerificationFailureAnalyzer()
    defn = FailureAnalysisDefinition()

    # Case 1: Low confidence (< 0.5) -> should trigger Verification Failure
    low_conf_result = VerificationResult(
        verdict=VerificationVerdict.SUPPORTED,
        confidence=0.3,
        evidence_bundle=dummy_evidence_bundle,
    )
    inp = FailureAnalysisInput(
        claim="Test",
        pipeline_artifacts={"verification_result": low_conf_result},
        definition=defn,
    )
    res = analyzer.analyze(inp)
    assert res.classification.category == FailureCategory.VERIFICATION
    assert res.classification.severity == FailureSeverity.HIGH
    assert res.diagnostic.root_cause == FailureRootCause.LOW_CONFIDENCE

    # Case 2: Contradictory evidence (both supporting and contradicting exist)
    contra_result = VerificationResult(
        verdict=VerificationVerdict.SUPPORTED,
        confidence=0.8,
        evidence_bundle=dummy_evidence_bundle,
        supporting_passages=("s1",),
        contradicting_passages=("s2",),
    )
    inp_contra = FailureAnalysisInput(
        claim="Test",
        pipeline_artifacts={"verification_result": contra_result},
        definition=defn,
    )
    res_contra = analyzer.analyze(inp_contra)
    assert res_contra.classification.category == FailureCategory.AGGREGATION
    assert res_contra.classification.severity == FailureSeverity.MEDIUM
    assert res_contra.diagnostic.root_cause == FailureRootCause.CONTRADICTORY_EVIDENCE


def test_calibration_failure_analyzer(
    dummy_verification_result: VerificationResult,
) -> None:
    analyzer = CalibrationFailureAnalyzer()
    defn = FailureAnalysisDefinition()

    # Case 1: Out of bounds calibrated confidence -> Calibration Failure
    trace = CalibrationTrace(
        original_confidence=0.9,
        intermediate_values={},
        final_confidence=1.5,
        applied_strategy=CalibrationStrategyType.IDENTITY,
        parameter_version="1.0",
    )
    cal_res = CalibrationResult(
        original_confidence=0.9,
        calibrated_confidence=1.5,  # Invalid confidence
        uncertainty_estimate=0.1,
        calibration_trace=trace,
    )
    inp = FailureAnalysisInput(
        claim="Test",
        pipeline_artifacts={
            "verification_result": dummy_verification_result,
            "calibration_result": cal_res,
        },
        definition=defn,
    )
    res = analyzer.analyze(inp)
    assert res.classification.category == FailureCategory.CALIBRATION
    assert res.classification.severity == FailureSeverity.HIGH
    assert res.diagnostic.root_cause == FailureRootCause.CALIBRATION_FAILURE


def test_infrastructure_failure_analyzer(
    dummy_verification_result: VerificationResult,
) -> None:
    analyzer = InfrastructureFailureAnalyzer()
    defn = FailureAnalysisDefinition()

    # Case 1: Execution duration exceeds threshold -> Infrastructure / Timeout Failure
    inp = FailureAnalysisInput(
        claim="Test",
        pipeline_artifacts={
            "verification_result": dummy_verification_result,
            "execution_duration": 6000.0,  # Exceeds 5000ms bounds
        },
        definition=defn,
    )
    res = analyzer.analyze(inp)
    assert res.classification.category == FailureCategory.INFRASTRUCTURE
    assert res.classification.severity == FailureSeverity.HIGH
    assert res.diagnostic.root_cause == FailureRootCause.TIMEOUT


def test_failure_aggregation_strategy(
    dummy_verification_result: VerificationResult,
) -> None:
    strategy = DefaultFailureAggregationStrategy()
    defn = FailureAnalysisDefinition()
    inp = FailureAnalysisInput(
        claim="Test",
        pipeline_artifacts={"verification_result": dummy_verification_result},
        definition=defn,
    )

    r_meta = FailureRuntimeMetadata(
        analyzer_id="a1",
        analyzer_version="1.0",
        execution_environment="prod",
        execution_device="cpu",
        framework="python",
        execution_timestamp="now",
    )

    # Prepare two mock outputs representing Verification (HIGH) and Configuration (CRITICAL)
    res1 = AnalyzerExecutionResult(
        analyzer_id="a1",
        execution_order=0,
        classification=FailureClassification(
            category=FailureCategory.VERIFICATION,
            severity=FailureSeverity.HIGH,
            affected_subsystem="verification",
        ),
        diagnostic_evidence=(
            DiagnosticEvidence(
                analyzer_id="a1",
                artifact_reference=FailureArtifactReference(
                    artifact_id="ver_res", artifact_type="out", subsystem="verification"
                ),
                detected_issue="Low confidence issue",
                confidence=1.0,
            ),
        ),
        runtime_metadata=r_meta,
    )

    res2 = AnalyzerExecutionResult(
        analyzer_id="a2",
        execution_order=1,
        classification=FailureClassification(
            category=FailureCategory.CONFIGURATION,
            severity=FailureSeverity.CRITICAL,
            affected_subsystem="configuration",
        ),
        diagnostic_evidence=(
            DiagnosticEvidence(
                analyzer_id="a2",
                artifact_reference=FailureArtifactReference(
                    artifact_id="conf_res",
                    artifact_type="conf",
                    subsystem="configuration",
                ),
                detected_issue="Incompatible configuration settings",
                confidence=1.0,
            ),
        ),
        runtime_metadata=r_meta,
    )

    aggregated = strategy.aggregate((res1, res2), inp)

    # CONFIGURATION is highest precedence category; CRITICAL is highest precedence severity
    assert aggregated.classification.category == FailureCategory.CONFIGURATION
    assert aggregated.classification.severity == FailureSeverity.CRITICAL
    assert aggregated.diagnostic.root_cause == FailureRootCause.CONFIGURATION_FAILURE
    assert "Low confidence issue" in aggregated.diagnostic.diagnostic_summary
    assert (
        "Incompatible configuration settings"
        in aggregated.diagnostic.diagnostic_summary
    )


def test_composite_failure_analyzer_ordering_and_determinism(
    dummy_verification_result: VerificationResult,
) -> None:
    defn = FailureAnalysisDefinition()
    inp = FailureAnalysisInput(
        claim="Test",
        pipeline_artifacts={"verification_result": dummy_verification_result},
        definition=defn,
    )

    composite = CompositeFailureAnalyzer(
        analyzers=(RetrievalFailureAnalyzer(), VerificationFailureAnalyzer()),
        aggregation_strategy=DefaultFailureAggregationStrategy(),
    )

    res1 = composite.analyze(inp)
    res2 = composite.analyze(inp)

    # Must be identical and deterministic
    assert res1.classification == res2.classification
    assert res1.diagnostic == res2.diagnostic
    assert res1.trace.analyzer_execution_order == (
        "retrieval_failure_analyzer",
        "verification_failure_analyzer",
    )


def test_bootstrap_building_and_validations() -> None:
    settings = Settings()
    registry = build_failure_analysis_registry(settings)
    profile = registry.resolve("default_failure_analysis")
    assert profile is not None

    # Verify that the configured analyzer is a CompositeFailureAnalyzer
    assert isinstance(profile.analyzer, CompositeFailureAnalyzer)


def test_legacy_compatibility(dummy_verification_result: VerificationResult) -> None:
    analyzer = DefaultFailureAnalyzer()
    defn = FailureAnalysisDefinition()

    # Trigger via legacy adapter
    with pytest.deprecated_call():
        res_legacy = analyzer.analyze("Claim", dummy_verification_result, defn)

    assert res_legacy.classification is not None
    assert res_legacy.diagnostic is not None


def test_failure_correlation_identity() -> None:
    from src.core.failure.correlation import DefaultFailureCorrelationStrategy
    from src.core.failure.failure_models import (
        AnalyzerExecutionResult,
        FailureCategory,
        FailureClassification,
        FailureCorrelationContext,
        FailureRuntimeMetadata,
        FailureSeverity,
    )

    r_meta = FailureRuntimeMetadata(
        analyzer_id="r1",
        analyzer_version="1.0",
        execution_environment="prod",
        execution_device="cpu",
        framework="python",
        execution_timestamp="now",
    )

    res = AnalyzerExecutionResult(
        analyzer_id="r1",
        execution_order=0,
        classification=FailureClassification(
            category=FailureCategory.RETRIEVAL,
            severity=FailureSeverity.CRITICAL,
            affected_subsystem="retrieval",
        ),
        diagnostic_evidence=(),
        runtime_metadata=r_meta,
    )

    strategy = DefaultFailureCorrelationStrategy()
    context = FailureCorrelationContext(
        analyzer_execution_results=(res,),
        correlation_rules=(),
    )

    correlated = strategy.correlate(context)
    # A single node with no rules produces no edges, but node is represented in the root list
    assert len(correlated.correlation_graph) == 0
    assert correlated.root_failures == ("r1",)


def test_failure_correlation_dependency_graph() -> None:
    from src.core.failure.correlation import DefaultFailureCorrelationStrategy
    from src.core.failure.failure_models import (
        AnalyzerExecutionResult,
        FailureCategory,
        FailureClassification,
        FailureCorrelationContext,
        FailureCorrelationRule,
        FailureRuntimeMetadata,
        FailureSeverity,
    )

    r_meta = FailureRuntimeMetadata(
        analyzer_id="r1",
        analyzer_version="1.0",
        execution_environment="prod",
        execution_device="cpu",
        framework="python",
        execution_timestamp="now",
    )

    # Retrieval failure
    res_ret = AnalyzerExecutionResult(
        analyzer_id="retrieval_analyzer",
        execution_order=0,
        classification=FailureClassification(
            category=FailureCategory.RETRIEVAL,
            severity=FailureSeverity.CRITICAL,
            affected_subsystem="retrieval",
        ),
        diagnostic_evidence=(),
        runtime_metadata=r_meta,
    )

    # Verification failure
    res_ver = AnalyzerExecutionResult(
        analyzer_id="verification_analyzer",
        execution_order=1,
        classification=FailureClassification(
            category=FailureCategory.VERIFICATION,
            severity=FailureSeverity.HIGH,
            affected_subsystem="verification",
        ),
        diagnostic_evidence=(),
        runtime_metadata=r_meta,
    )

    # Calibration failure
    res_cal = AnalyzerExecutionResult(
        analyzer_id="calibration_analyzer",
        execution_order=2,
        classification=FailureClassification(
            category=FailureCategory.CALIBRATION,
            severity=FailureSeverity.HIGH,
            affected_subsystem="calibration",
        ),
        diagnostic_evidence=(),
        runtime_metadata=r_meta,
    )

    # Rules: Retrieval -> Verification, Verification -> Calibration
    rule1 = FailureCorrelationRule(
        rule_id="rule1",
        source_category=FailureCategory.RETRIEVAL,
        target_category=FailureCategory.VERIFICATION,
        precedence=1,
        enabled=True,
    )
    rule2 = FailureCorrelationRule(
        rule_id="rule2",
        source_category=FailureCategory.VERIFICATION,
        target_category=FailureCategory.CALIBRATION,
        precedence=1,
        enabled=True,
    )

    strategy = DefaultFailureCorrelationStrategy()
    context = FailureCorrelationContext(
        analyzer_execution_results=(res_ret, res_ver, res_cal),
        correlation_rules=(rule1, rule2),
    )

    correlated = strategy.correlate(context)
    assert len(correlated.correlation_graph) == 2
    # root failure should be retrieval_analyzer since verification_analyzer and calibration_analyzer have in-degree > 0
    assert correlated.root_failures == ("retrieval_analyzer",)
    assert correlated.dependency_edges["retrieval_analyzer"] == (
        "verification_analyzer",
    )
    assert correlated.dependency_edges["verification_analyzer"] == (
        "calibration_analyzer",
    )


def test_failure_correlation_independent_failures() -> None:
    from src.core.failure.correlation import DefaultFailureCorrelationStrategy
    from src.core.failure.failure_models import (
        AnalyzerExecutionResult,
        FailureCategory,
        FailureClassification,
        FailureCorrelationContext,
        FailureCorrelationRule,
        FailureRuntimeMetadata,
        FailureSeverity,
    )

    r_meta = FailureRuntimeMetadata(
        analyzer_id="r1",
        analyzer_version="1.0",
        execution_environment="prod",
        execution_device="cpu",
        framework="python",
        execution_timestamp="now",
    )

    res_ret = AnalyzerExecutionResult(
        analyzer_id="retrieval_analyzer",
        execution_order=0,
        classification=FailureClassification(
            category=FailureCategory.RETRIEVAL,
            severity=FailureSeverity.CRITICAL,
            affected_subsystem="retrieval",
        ),
        diagnostic_evidence=(),
        runtime_metadata=r_meta,
    )

    res_cal = AnalyzerExecutionResult(
        analyzer_id="calibration_analyzer",
        execution_order=1,
        classification=FailureClassification(
            category=FailureCategory.CALIBRATION,
            severity=FailureSeverity.HIGH,
            affected_subsystem="calibration",
        ),
        diagnostic_evidence=(),
        runtime_metadata=r_meta,
    )

    # Rule (unmatched): Verification -> Calibration
    rule1 = FailureCorrelationRule(
        rule_id="rule1",
        source_category=FailureCategory.VERIFICATION,
        target_category=FailureCategory.CALIBRATION,
        precedence=1,
        enabled=True,
    )

    strategy = DefaultFailureCorrelationStrategy()
    context = FailureCorrelationContext(
        analyzer_execution_results=(res_ret, res_cal),
        correlation_rules=(rule1,),
    )

    correlated = strategy.correlate(context)
    # No matches, so both are independent roots
    assert len(correlated.correlation_graph) == 0
    assert "retrieval_analyzer" in correlated.root_failures
    assert "calibration_analyzer" in correlated.root_failures


def test_failure_correlation_registries_and_bootstrap() -> None:
    from src.core.bootstrap import build_failure_correlation_registry
    from src.core.failure.correlation import DefaultFailureCorrelationStrategy
    from src.core.failure.failure_models import (
        FailureCategory,
        FailureCorrelationDefinition,
        FailureCorrelationProfile,
        FailureCorrelationProfileRegistry,
        FailureCorrelationRule,
    )

    # Test bootstrap builder
    settings = Settings()
    registry = build_failure_correlation_registry(settings)
    profile = registry.resolve("default_failure_correlation")
    assert profile is not None
    assert isinstance(profile.strategy, DefaultFailureCorrelationStrategy)
    assert len(profile.rules) == 3

    # Test duplicate rule detection
    rule1 = FailureCorrelationRule(
        rule_id="dup_rule",
        source_category=FailureCategory.RETRIEVAL,
        target_category=FailureCategory.VERIFICATION,
    )
    profile_dup = FailureCorrelationProfile(
        profile_id="dup_profile",
        definition=FailureCorrelationDefinition(),
        rules=(rule1, rule1),
        strategy=DefaultFailureCorrelationStrategy(),
    )
    from src.core.exceptions import DuplicateFailureAnalysisProfileError

    with pytest.raises(DuplicateFailureAnalysisProfileError):
        FailureCorrelationProfileRegistry(profiles=(profile_dup,))


# ===========================================================================
# M3.5 Root Cause Attribution & Severity Policies
# ===========================================================================


def _make_correlation_result_chain() -> "Any":
    """Helper: build a Retrieval -> Verification -> Calibration correlation chain."""
    from src.core.failure.failure_models import (
        FailureCorrelation,
        FailureCorrelationResult,
    )

    edge1 = FailureCorrelation(
        correlation_id="e1",
        source_failure="retrieval_node",
        target_failure="verification_node",
        correlation_confidence=1.0,
    )
    edge2 = FailureCorrelation(
        correlation_id="e2",
        source_failure="verification_node",
        target_failure="calibration_node",
        correlation_confidence=1.0,
    )
    return FailureCorrelationResult(
        correlation_graph=(edge1, edge2),
        root_failures=("retrieval_node",),
        dependency_edges={
            "retrieval_node": ("verification_node",),
            "verification_node": ("calibration_node",),
            "calibration_node": (),
        },
        summary="test chain",
    )


def test_traverser_root_detection() -> None:
    from src.core.failure.traversal import FailureGraphTraverser

    result = _make_correlation_result_chain()
    traverser = FailureGraphTraverser()
    roots = traverser.get_root_nodes(result)
    assert roots == ("retrieval_node",)


def test_traverser_downstream_discovery() -> None:
    from src.core.failure.traversal import FailureGraphTraverser

    result = _make_correlation_result_chain()
    traverser = FailureGraphTraverser()
    downstream = traverser.get_downstream_nodes("retrieval_node", result)
    assert downstream == ("verification_node",)


def test_traverser_all_descendants() -> None:
    from src.core.failure.traversal import FailureGraphTraverser

    result = _make_correlation_result_chain()
    traverser = FailureGraphTraverser()
    descendants = traverser.get_all_descendants("retrieval_node", result)
    assert "verification_node" in descendants
    assert "calibration_node" in descendants


def test_traverser_dependency_path() -> None:
    from src.core.failure.traversal import FailureGraphTraverser

    result = _make_correlation_result_chain()
    traverser = FailureGraphTraverser()
    path = traverser.build_dependency_path("retrieval_node", result)
    assert path == ("retrieval_node", "verification_node", "calibration_node")


def test_root_cause_single_root() -> None:
    from src.core.failure.attribution import DependencyGraphRootCauseStrategy
    from src.core.failure.failure_models import RootCauseAttributionDefinition
    from src.core.failure.traversal import FailureGraphTraverser

    result = _make_correlation_result_chain()
    strategy = DependencyGraphRootCauseStrategy()
    definition = RootCauseAttributionDefinition()
    traverser = FailureGraphTraverser()

    rc = strategy.attribute(result, definition, traverser)
    assert rc.primary_root_cause == "retrieval_node"
    assert (
        "verification_node" in rc.contributing_failures
        or "calibration_node" in rc.contributing_failures
    )
    assert rc.dependency_path[0] == "retrieval_node"


def test_root_cause_multiple_roots_deterministic_precedence() -> None:
    from src.core.failure.attribution import DependencyGraphRootCauseStrategy
    from src.core.failure.failure_models import (
        FailureCorrelationResult,
        RootCauseAttributionDefinition,
    )
    from src.core.failure.traversal import FailureGraphTraverser

    # Two independent roots: alphabetically "aaa_node" comes first.
    result = FailureCorrelationResult(
        correlation_graph=(),
        root_failures=("zzz_node", "aaa_node"),
        dependency_edges={"aaa_node": (), "zzz_node": ()},
        summary="two independent roots",
    )

    strategy = DependencyGraphRootCauseStrategy()
    definition = RootCauseAttributionDefinition()
    traverser = FailureGraphTraverser()

    rc = strategy.attribute(result, definition, traverser)
    # Alphabetical tie-breaking: aaa_node < zzz_node
    assert rc.primary_root_cause == "aaa_node"
    assert "zzz_node" in rc.contributing_failures


def test_root_cause_downstream_filtered_as_contributing() -> None:
    from src.core.failure.attribution import DependencyGraphRootCauseStrategy
    from src.core.failure.failure_models import RootCauseAttributionDefinition
    from src.core.failure.traversal import FailureGraphTraverser

    result = _make_correlation_result_chain()
    strategy = DependencyGraphRootCauseStrategy()
    definition = RootCauseAttributionDefinition()
    traverser = FailureGraphTraverser()

    rc = strategy.attribute(result, definition, traverser)
    # verification_node and calibration_node are downstream — contributing, not primary.
    assert rc.primary_root_cause != "verification_node"
    assert rc.primary_root_cause != "calibration_node"


def test_root_cause_determinism() -> None:
    from src.core.failure.attribution import DependencyGraphRootCauseStrategy
    from src.core.failure.failure_models import RootCauseAttributionDefinition
    from src.core.failure.traversal import FailureGraphTraverser

    result = _make_correlation_result_chain()
    strategy = DependencyGraphRootCauseStrategy()
    definition = RootCauseAttributionDefinition()
    traverser = FailureGraphTraverser()

    rc1 = strategy.attribute(result, definition, traverser)
    rc2 = strategy.attribute(result, definition, traverser)
    assert rc1 == rc2


def test_severity_rule_evaluation() -> None:
    from src.core.failure.failure_models import (
        FailureCategory,
        FailureSeverity,
        RootCauseResult,
        SeverityPolicyDefinition,
        SeverityRule,
    )
    from src.core.failure.severity import ThresholdSeverityPolicy

    rule = SeverityRule(
        rule_id="infra_critical",
        category=FailureCategory.INFRASTRUCTURE,
        minimum_confidence=0.0,
        severity=FailureSeverity.CRITICAL,
        escalation_required=True,
        priority=1,
    )
    definition = SeverityPolicyDefinition(rules=(rule,))
    policy = ThresholdSeverityPolicy()

    rc = RootCauseResult(
        primary_root_cause="infra_node",
        contributing_failures=(),
        dependency_path=("infra_node",),
        attribution_confidence=1.0,
    )

    result = policy.evaluate(rc, definition)
    assert result.overall_severity == FailureSeverity.CRITICAL
    assert result.escalation_required is True
    assert result.applied_rule == "infra_critical"


def test_severity_escalation_policy() -> None:
    from src.core.failure.failure_models import (
        FailureCategory,
        FailureSeverity,
        RootCauseResult,
        SeverityPolicyDefinition,
        SeverityRule,
    )
    from src.core.failure.severity import ThresholdSeverityPolicy

    rule = SeverityRule(
        rule_id="retrieval_high",
        category=FailureCategory.RETRIEVAL,
        minimum_confidence=0.0,
        severity=FailureSeverity.HIGH,
        escalation_required=False,
        priority=1,
    )
    definition = SeverityPolicyDefinition(rules=(rule,))
    policy = ThresholdSeverityPolicy()

    rc = RootCauseResult(
        primary_root_cause="retrieval_node",
        contributing_failures=(),
        dependency_path=("retrieval_node",),
        attribution_confidence=1.0,
    )
    result = policy.evaluate(rc, definition)
    assert result.overall_severity == FailureSeverity.HIGH
    assert result.escalation_required is False


def test_severity_category_override() -> None:
    from src.core.failure.failure_models import (
        FailureCategory,
        FailureSeverity,
        RootCauseResult,
        SeverityPolicyDefinition,
        SeverityRule,
    )
    from src.core.failure.severity import ThresholdSeverityPolicy

    rule = SeverityRule(
        rule_id="retrieval_rule",
        category=FailureCategory.RETRIEVAL,
        minimum_confidence=0.0,
        severity=FailureSeverity.HIGH,
        escalation_required=False,
        priority=1,
    )
    # Override: RETRIEVAL should map to CRITICAL instead.
    definition = SeverityPolicyDefinition(
        rules=(rule,),
        category_overrides={"RETRIEVAL": FailureSeverity.CRITICAL},
    )
    policy = ThresholdSeverityPolicy()

    rc = RootCauseResult(
        primary_root_cause="retrieval_node",
        contributing_failures=(),
        dependency_path=("retrieval_node",),
        attribution_confidence=1.0,
    )
    result = policy.evaluate(rc, definition)
    # Override takes precedence.
    assert result.overall_severity == FailureSeverity.CRITICAL


def test_severity_default_when_no_rules_match() -> None:
    from src.core.failure.failure_models import (
        FailureCategory,
        FailureSeverity,
        RootCauseResult,
        SeverityPolicyDefinition,
        SeverityRule,
    )
    from src.core.failure.severity import ThresholdSeverityPolicy

    rule = SeverityRule(
        rule_id="infra_rule",
        category=FailureCategory.INFRASTRUCTURE,
        minimum_confidence=0.99,  # High threshold — will not match confidence=0.1
        severity=FailureSeverity.CRITICAL,
        escalation_required=True,
        priority=1,
    )
    definition = SeverityPolicyDefinition(
        rules=(rule,),
        default_severity=FailureSeverity.LOW,
    )
    policy = ThresholdSeverityPolicy()

    rc = RootCauseResult(
        primary_root_cause="infra_node",
        contributing_failures=(),
        dependency_path=("infra_node",),
        attribution_confidence=0.1,  # Below minimum_confidence
    )
    result = policy.evaluate(rc, definition)
    assert result.overall_severity == FailureSeverity.LOW
    assert result.applied_rule is None


def test_severity_determinism() -> None:
    from src.core.failure.failure_models import (
        FailureCategory,
        FailureSeverity,
        RootCauseResult,
        SeverityPolicyDefinition,
        SeverityRule,
    )
    from src.core.failure.severity import ThresholdSeverityPolicy

    rule = SeverityRule(
        rule_id="r1",
        category=FailureCategory.RETRIEVAL,
        minimum_confidence=0.0,
        severity=FailureSeverity.HIGH,
        escalation_required=False,
        priority=1,
    )
    definition = SeverityPolicyDefinition(rules=(rule,))
    policy = ThresholdSeverityPolicy()
    rc = RootCauseResult(
        primary_root_cause="retrieval_node",
        contributing_failures=(),
        dependency_path=("retrieval_node",),
        attribution_confidence=1.0,
    )
    result1 = policy.evaluate(rc, definition)
    result2 = policy.evaluate(rc, definition)
    assert result1 == result2


def test_root_cause_registry_duplicate_detection() -> None:
    from src.core.exceptions import DuplicateFailureAnalysisProfileError
    from src.core.failure.attribution import DependencyGraphRootCauseStrategy
    from src.core.failure.failure_models import (
        RootCauseAttributionDefinition,
        RootCauseProfile,
        RootCauseProfileRegistry,
    )

    strategy = DependencyGraphRootCauseStrategy()
    definition = RootCauseAttributionDefinition()
    profile = RootCauseProfile(
        profile_id="dup_rc",
        definition=definition,
        strategy=strategy,
    )
    with pytest.raises(DuplicateFailureAnalysisProfileError):
        RootCauseProfileRegistry(profiles=(profile, profile))


def test_severity_registry_duplicate_detection() -> None:
    from src.core.exceptions import DuplicateFailureAnalysisProfileError
    from src.core.failure.failure_models import (
        SeverityPolicyDefinition,
        SeverityPolicyProfile,
        SeverityPolicyRegistry,
    )
    from src.core.failure.severity import ThresholdSeverityPolicy

    policy = ThresholdSeverityPolicy()
    definition = SeverityPolicyDefinition()
    profile = SeverityPolicyProfile(
        profile_id="dup_sev",
        definition=definition,
        policy=policy,
    )
    with pytest.raises(DuplicateFailureAnalysisProfileError):
        SeverityPolicyRegistry(profiles=(profile, profile))


def test_bootstrap_root_cause_registry() -> None:
    from src.core.bootstrap import build_root_cause_registry
    from src.core.failure.attribution import DependencyGraphRootCauseStrategy

    settings = Settings()
    registry = build_root_cause_registry(settings)
    profile = registry.resolve("default_root_cause")
    assert profile is not None
    assert isinstance(profile.strategy, DependencyGraphRootCauseStrategy)


def test_bootstrap_severity_policy_registry() -> None:
    from src.core.bootstrap import build_severity_policy_registry
    from src.core.failure.severity import ThresholdSeverityPolicy

    settings = Settings()
    registry = build_severity_policy_registry(settings)
    profile = registry.resolve("default_severity_policy")
    assert profile is not None
    assert isinstance(profile.policy, ThresholdSeverityPolicy)
    assert len(profile.definition.rules) == 5


def test_m35_end_to_end_integration() -> None:
    """FailureCorrelationResult -> Traverser -> Attribution -> Severity."""
    from src.core.failure.attribution import DependencyGraphRootCauseStrategy
    from src.core.failure.failure_models import (
        FailureCategory,
        FailureSeverity,
        RootCauseAttributionDefinition,
        SeverityPolicyDefinition,
        SeverityRule,
    )
    from src.core.failure.severity import ThresholdSeverityPolicy
    from src.core.failure.traversal import FailureGraphTraverser

    correlation_result = _make_correlation_result_chain()
    traverser = FailureGraphTraverser()
    strategy = DependencyGraphRootCauseStrategy()
    rc_definition = RootCauseAttributionDefinition()

    root_cause = strategy.attribute(correlation_result, rc_definition, traverser)
    assert root_cause.primary_root_cause == "retrieval_node"

    sev_rule = SeverityRule(
        rule_id="retrieval_high",
        category=FailureCategory.RETRIEVAL,
        minimum_confidence=0.0,
        severity=FailureSeverity.HIGH,
        escalation_required=False,
        priority=1,
    )
    sev_definition = SeverityPolicyDefinition(rules=(sev_rule,))
    sev_policy = ThresholdSeverityPolicy()

    sev_result = sev_policy.evaluate(root_cause, sev_definition)
    assert sev_result.overall_severity == FailureSeverity.HIGH
    assert sev_result.applied_rule == "retrieval_high"


# ===========================================================================
# M3.6 Failure Benchmarking & Evaluation Framework
# ===========================================================================


def _make_benchmark_item(
    item_id: str = "item1",
    expected_category: "Any" = None,
    expected_root_cause: "Any" = None,
    expected_severity: "Any" = None,
) -> "Any":
    """Helper: build a FailureBenchmarkItem with real analyzer results."""
    from src.core.failure.benchmark.benchmark_models import FailureBenchmarkItem
    from src.core.failure.failure_models import (
        AnalyzerExecutionResult,
        FailureCategory,
        FailureClassification,
        FailureRootCause,
        FailureRuntimeMetadata,
        FailureSeverity,
    )

    expected_category = expected_category or FailureCategory.RETRIEVAL
    expected_root_cause = expected_root_cause or FailureRootCause.LOW_RETRIEVAL_RECALL
    expected_severity = expected_severity or FailureSeverity.HIGH

    r_meta = FailureRuntimeMetadata(
        analyzer_id="test_analyzer",
        analyzer_version="1.0",
        execution_environment="test",
        execution_device="cpu",
        framework="python",
        execution_timestamp="now",
    )

    exec_result = AnalyzerExecutionResult(
        analyzer_id="test_analyzer",
        execution_order=0,
        classification=FailureClassification(
            category=expected_category,
            severity=expected_severity,
            affected_subsystem="retrieval",
        ),
        diagnostic_evidence=(),
        runtime_metadata=r_meta,
    )

    return FailureBenchmarkItem(
        item_id=item_id,
        analyzer_execution_results=(exec_result,),
        expected_category=expected_category,
        expected_root_cause=expected_root_cause,
        expected_severity=expected_severity,
    )


def _make_simple_dataset(items: "Any") -> "Any":
    """In-memory dataset for testing."""

    class _InMemoryDataset:
        @property
        def name(self) -> str:
            return "test_dataset"

        @property
        def version(self) -> str:
            return "1.0"

        @property
        def description(self) -> str:
            return "Test-only synthetic dataset."

        def items(self) -> tuple[Any, ...]:
            return tuple(items)

    return _InMemoryDataset()


def test_benchmark_item_valid_construction() -> None:
    item = _make_benchmark_item()
    assert item.item_id == "item1"
    assert item.analyzer_execution_results is not None


def test_benchmark_item_invalid_construction() -> None:
    from pydantic import ValidationError

    from src.core.failure.benchmark.benchmark_models import FailureBenchmarkItem
    from src.core.failure.failure_models import (
        FailureCategory,
        FailureRootCause,
        FailureSeverity,
    )

    with pytest.raises(ValidationError):
        FailureBenchmarkItem(
            item_id="",  # violates min_length=1
            analyzer_execution_results=(),
            expected_category=FailureCategory.RETRIEVAL,
            expected_root_cause=FailureRootCause.UNKNOWN,
            expected_severity=FailureSeverity.INFO,
        )


def test_metric_engine_accuracy() -> None:
    from src.core.failure.benchmark.metrics import (
        AccuracyCalculator,
        FailureBenchmarkRawOutput,
        FailureMetricEngine,
    )
    from src.core.failure.failure_models import (
        FailureCategory,
        FailureRootCause,
        FailureSeverity,
    )

    item = _make_benchmark_item(expected_category=FailureCategory.RETRIEVAL)
    raw = FailureBenchmarkRawOutput()
    raw.items.append(item)
    raw.actual_categories.append(FailureCategory.RETRIEVAL)  # correct
    raw.actual_root_causes.append(FailureRootCause.UNKNOWN)
    raw.actual_severities.append(FailureSeverity.HIGH)
    raw.latencies_ms.append(1.5)
    raw.repeated_categories.append([FailureCategory.RETRIEVAL])

    acc = AccuracyCalculator().calculate(raw)
    assert acc == 1.0

    engine = FailureMetricEngine()
    result = engine.compute(raw, ())
    assert result.metric_values["classification_accuracy"] == 1.0


def test_metric_engine_attribution_accuracy() -> None:
    from src.core.failure.benchmark.metrics import (
        AttributionAccuracyCalculator,
        FailureBenchmarkRawOutput,
    )
    from src.core.failure.failure_models import (
        FailureCategory,
        FailureRootCause,
        FailureSeverity,
    )

    item = _make_benchmark_item(
        expected_root_cause=FailureRootCause.LOW_RETRIEVAL_RECALL
    )
    raw = FailureBenchmarkRawOutput()
    raw.items.append(item)
    raw.actual_categories.append(FailureCategory.RETRIEVAL)
    raw.actual_root_causes.append(FailureRootCause.LOW_RETRIEVAL_RECALL)  # correct
    raw.actual_severities.append(FailureSeverity.HIGH)
    raw.latencies_ms.append(2.0)

    calc = AttributionAccuracyCalculator()
    assert calc.calculate(raw) == 1.0


def test_metric_engine_severity_consistency() -> None:
    from src.core.failure.benchmark.metrics import (
        FailureBenchmarkRawOutput,
        SeverityConsistencyCalculator,
    )
    from src.core.failure.failure_models import (
        FailureCategory,
        FailureRootCause,
        FailureSeverity,
    )

    item = _make_benchmark_item(expected_severity=FailureSeverity.HIGH)
    raw = FailureBenchmarkRawOutput()
    raw.items.append(item)
    raw.actual_categories.append(FailureCategory.RETRIEVAL)
    raw.actual_root_causes.append(FailureRootCause.UNKNOWN)
    raw.actual_severities.append(FailureSeverity.HIGH)  # matches expected
    raw.latencies_ms.append(1.0)

    assert SeverityConsistencyCalculator().calculate(raw) == 1.0


def test_metric_engine_latency_metrics() -> None:
    from src.core.failure.benchmark.metrics import (
        FailureBenchmarkRawOutput,
        MeanLatencyCalculator,
        P95LatencyCalculator,
        P99LatencyCalculator,
    )
    from src.core.failure.failure_models import (
        FailureCategory,
        FailureRootCause,
        FailureSeverity,
    )

    raw = FailureBenchmarkRawOutput()
    for i in range(10):
        item = _make_benchmark_item(item_id=f"item{i}")
        raw.items.append(item)
        raw.actual_categories.append(FailureCategory.RETRIEVAL)
        raw.actual_root_causes.append(FailureRootCause.UNKNOWN)
        raw.actual_severities.append(FailureSeverity.HIGH)
        raw.latencies_ms.append(float(i + 1))

    mean = MeanLatencyCalculator().calculate(raw)
    assert abs(mean - 5.5) < 0.01

    p95 = P95LatencyCalculator().calculate(raw)
    assert p95 >= 9.0

    p99 = P99LatencyCalculator().calculate(raw)
    assert p99 >= 9.0


def test_benchmark_runner_single_item() -> None:
    from src.core.failure.benchmark.benchmark_models import (
        FailureBenchmarkDefinition,
        FailureBenchmarkSuite,
    )
    from src.core.failure.benchmark.runner import FailureBenchmarkRunner

    item = _make_benchmark_item()
    dataset = _make_simple_dataset([item])

    suite = FailureBenchmarkSuite(
        suite_id="test_suite",
        dataset=dataset,
        enabled_metrics=(),
        evaluation_profile="default",
    )
    definition = FailureBenchmarkDefinition(determinism_runs=1)
    runner = FailureBenchmarkRunner()

    report = runner.run(suite, definition)
    assert report.configuration_fingerprint
    assert report.execution_timestamp
    assert "classification_accuracy" in report.result.metric_values


def test_benchmark_runner_multiple_items() -> None:
    from src.core.failure.benchmark.benchmark_models import (
        FailureBenchmarkDefinition,
        FailureBenchmarkSuite,
    )
    from src.core.failure.benchmark.runner import FailureBenchmarkRunner

    items = [_make_benchmark_item(item_id=f"item{i}") for i in range(5)]
    dataset = _make_simple_dataset(items)

    suite = FailureBenchmarkSuite(
        suite_id="multi_suite",
        dataset=dataset,
        enabled_metrics=(),
        evaluation_profile="default",
    )
    definition = FailureBenchmarkDefinition(determinism_runs=2)
    runner = FailureBenchmarkRunner()

    report = runner.run(suite, definition)
    assert report.result.execution_metadata["item_count"] == 5
    assert report.result.metric_values["classification_accuracy"] == 1.0


def test_benchmark_determinism() -> None:
    from src.core.failure.benchmark.benchmark_models import (
        FailureBenchmarkDefinition,
        FailureBenchmarkSuite,
    )
    from src.core.failure.benchmark.runner import FailureBenchmarkRunner

    item = _make_benchmark_item()
    dataset = _make_simple_dataset([item])
    suite = FailureBenchmarkSuite(
        suite_id="det_suite",
        dataset=dataset,
        enabled_metrics=(),
        evaluation_profile="default",
    )
    definition = FailureBenchmarkDefinition(determinism_runs=3)
    runner = FailureBenchmarkRunner()

    report1 = runner.run(suite, definition)
    report2 = runner.run(suite, definition)

    # Configuration fingerprint must be identical across runs.
    assert report1.configuration_fingerprint == report2.configuration_fingerprint

    # Logic-driven metrics (non-latency) must be identical across runs.
    _latency_metrics = {
        "mean_latency_ms",
        "p95_latency_ms",
        "p99_latency_ms",
        "throughput_items_per_sec",
    }
    logic_values_1 = {
        k: v
        for k, v in report1.result.metric_values.items()
        if k not in _latency_metrics
    }
    logic_values_2 = {
        k: v
        for k, v in report2.result.metric_values.items()
        if k not in _latency_metrics
    }
    assert logic_values_1 == logic_values_2


def test_benchmark_registry_duplicate_detection() -> None:
    from src.core.exceptions import DuplicateFailureAnalysisProfileError
    from src.core.failure.benchmark.benchmark_models import (
        FailureBenchmarkDefinition,
        FailureBenchmarkProfile,
        FailureBenchmarkProfileRegistry,
    )
    from src.core.failure.benchmark.runner import FailureBenchmarkRunner

    runner = FailureBenchmarkRunner()
    definition = FailureBenchmarkDefinition()
    profile = FailureBenchmarkProfile(
        profile_id="dup_bench",
        definition=definition,
        runner=runner,
    )
    with pytest.raises(DuplicateFailureAnalysisProfileError):
        FailureBenchmarkProfileRegistry(profiles=(profile, profile))


def test_benchmark_bootstrap_registry() -> None:
    from src.core.bootstrap import build_failure_benchmark_registry
    from src.core.failure.benchmark.runner import FailureBenchmarkRunner

    settings = Settings()
    registry = build_failure_benchmark_registry(settings)
    profile = registry.resolve("default_failure_benchmark")
    assert profile is not None
    assert isinstance(profile.runner, FailureBenchmarkRunner)


def test_m36_end_to_end_integration() -> None:
    """FailureBenchmarkSuite -> Runner -> MetricEngine -> FailureBenchmarkReport."""
    from src.core.failure.benchmark.benchmark_models import (
        FailureBenchmarkDefinition,
        FailureBenchmarkSuite,
    )
    from src.core.failure.benchmark.metrics import FailureMetricEngine
    from src.core.failure.benchmark.runner import FailureBenchmarkRunner

    items = [_make_benchmark_item(item_id=f"item{i}") for i in range(3)]
    dataset = _make_simple_dataset(items)

    suite = FailureBenchmarkSuite(
        suite_id="e2e_suite",
        dataset=dataset,
        enabled_metrics=(
            "classification_accuracy",
            "attribution_accuracy",
            "mean_latency_ms",
        ),
        evaluation_profile="default",
    )
    definition = FailureBenchmarkDefinition(determinism_runs=2)
    engine = FailureMetricEngine()
    runner = FailureBenchmarkRunner(metric_engine=engine)

    report = runner.run(suite, definition)
    assert report.result.metric_values["classification_accuracy"] == 1.0
    assert (
        report.result.metric_values["attribution_accuracy"] == 0.0
    )  # UNKNOWN != LOW_RETRIEVAL_RECALL
    assert report.result.metric_values["mean_latency_ms"] >= 0.0
    assert len(report.benchmark_trace) >= 3  # suite + items + items_processed
