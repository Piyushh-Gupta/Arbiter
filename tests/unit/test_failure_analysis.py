"""Unit tests for M3.2 Failure Analyzer Interfaces & Immutable Models."""

import pytest
from pydantic import ValidationError

from src.core.bootstrap import build_failure_analysis_registry
from src.core.config import Settings
from src.core.exceptions import DuplicateFailureAnalysisProfileError
from src.core.failure.failure_models import (
    FailureAnalysisDefinition,
    FailureAnalysisInput,
    FailureAnalysisProfile,
    FailureAnalysisProfileRegistry,
    FailureArtifactReference,
    FailureCategory,
    FailureDiagnosticContext,
    FailureExecutionMetadata,
    FailureRuntimeMetadata,
)
from src.core.failure.implementations import DefaultFailureAnalyzer
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
    # 1. FailureArtifactReference
    ref = FailureArtifactReference(
        artifact_id="a1",
        artifact_type="bundle",
        subsystem="retrieval",
    )
    with pytest.raises(ValidationError):
        setattr(ref, "subsystem", "verification")

    # 2. FailureRuntimeMetadata
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

    # 3. FailureExecutionMetadata
    exec_meta = FailureExecutionMetadata(
        request_id="req1",
        execution_duration=12.5,
        analyzer_profile="default",
        configuration_fingerprint="abc",
    )
    with pytest.raises(ValidationError):
        setattr(exec_meta, "execution_duration", 15.0)

    # 4. FailureDiagnosticContext
    context = FailureDiagnosticContext(
        ordered_analyzer_outputs=(),
        inspected_artifact_references=(ref,),
    )
    with pytest.raises(ValidationError):
        setattr(context, "ordered_analyzer_outputs", ("out",))

    # 5. FailureAnalysisInput
    defn = FailureAnalysisDefinition()
    inp = FailureAnalysisInput(
        claim="Claim text",
        pipeline_artifacts={"verification_result": "dummy"},
        definition=defn,
    )
    with pytest.raises(ValidationError):
        setattr(inp, "claim", "New claim")


def test_analyzer_compatibility_and_validation() -> None:
    defn = FailureAnalysisDefinition()
    analyzer = DefaultFailureAnalyzer()

    # Verify that analyzer has the expected metadata and category enums
    assert isinstance(analyzer.runtime_metadata, FailureRuntimeMetadata)
    assert FailureCategory.VERIFICATION in analyzer.supported_categories

    profile = FailureAnalysisProfile(
        profile_id="p1",
        definition=defn,
        analyzer=analyzer,
    )
    assert profile.profile_id == "p1"


def test_legacy_adapter_equivalence(
    dummy_verification_result: VerificationResult,
) -> None:
    analyzer = DefaultFailureAnalyzer()
    defn = FailureAnalysisDefinition()

    # Legacy invocation path (should trigger DeprecationWarning)
    with pytest.deprecated_call():
        res_legacy = analyzer.analyze("Claim", dummy_verification_result, defn)

    # Canonical invocation path
    input_data = FailureAnalysisInput(
        claim="Claim",
        pipeline_artifacts={"verification_result": dummy_verification_result},
        definition=defn,
    )
    res_canonical = analyzer.analyze(input_data)

    # Validate identical results
    assert res_legacy.classification == res_canonical.classification
    assert res_legacy.diagnostic == res_canonical.diagnostic
    assert (
        res_legacy.trace.analyzer_execution_order
        == res_canonical.trace.analyzer_execution_order
    )


def test_determinism_and_integration(
    dummy_verification_result: VerificationResult,
) -> None:
    analyzer = DefaultFailureAnalyzer()
    defn = FailureAnalysisDefinition()
    input_data = FailureAnalysisInput(
        claim="Claim text",
        pipeline_artifacts={"verification_result": dummy_verification_result},
        definition=defn,
    )

    res1 = analyzer.analyze(input_data)
    res2 = analyzer.analyze(input_data)

    assert res1.classification == res2.classification
    assert res1.diagnostic == res2.diagnostic
    assert (
        res1.trace.execution_metadata["verification_result"]
        == dummy_verification_result
    )


def test_registry_resolutions() -> None:
    defn = FailureAnalysisDefinition()
    analyzer = DefaultFailureAnalyzer()
    profile = FailureAnalysisProfile(
        profile_id="p_default",
        definition=defn,
        analyzer=analyzer,
    )

    registry = FailureAnalysisProfileRegistry(profiles=(profile,))
    assert registry.resolve("p_default") is profile

    with pytest.raises(DuplicateFailureAnalysisProfileError):
        FailureAnalysisProfileRegistry(profiles=(profile, profile))


def test_bootstrap_building() -> None:
    settings = Settings()
    registry = build_failure_analysis_registry(settings)
    assert registry.resolve("default_failure_analysis") is not None
