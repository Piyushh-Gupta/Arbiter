"""Unit tests for M3.3 Diagnostic Engines."""

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
