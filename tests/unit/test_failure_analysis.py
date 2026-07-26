"""Unit tests for the M10.1 Failure Analysis Framework."""

import pytest
from pydantic import ValidationError

from src.core.exceptions import (
    FailureAnalysisConfigurationError,
    FailureAnalysisExecutionError,
)
from src.core.failure_analysis.analyzer import FailureAnalyzer
from src.core.failure_analysis.base import BaseFailureAnalyzer
from src.core.failure_analysis.failure_analysis_models import (
    FailureAnalysisDefinition,
    FailureAnalysisResult,
    FailureFlag,
    FailureMetadata,
    FailureSeverity,
)
from src.core.retrieval.retrieval_models import EvidenceBundle, RetrievalMetadata
from src.core.verification.verification_models import (
    VerificationLabel,
    VerificationMetadata,
    VerificationResult,
)


class DummyDefinition(FailureAnalysisDefinition):
    pass


class MockAnalyzer(BaseFailureAnalyzer):
    def __init__(self, reject: bool = False, fail_analyze: bool = False):
        self.reject = reject
        self.fail_analyze = fail_analyze
        self.called_analyze = False

    def validate_compatibility(self, definition: FailureAnalysisDefinition) -> None:
        if self.reject:
            raise FailureAnalysisConfigurationError("Incompatible definition")

    def analyze(
        self,
        claim: str,
        verification_result: VerificationResult,
        definition: FailureAnalysisDefinition,
    ) -> FailureAnalysisResult:
        self.called_analyze = True
        if self.fail_analyze:
            raise FailureAnalysisExecutionError("Runtime failure")

        flag = FailureFlag(code="TEST_FLAG", description="Test flag description")
        return FailureAnalysisResult(
            failure_flags=frozenset({flag}),
            severity=FailureSeverity.LOW,
            verification_result=verification_result,
            metadata=FailureMetadata(strategy_id="mock"),
        )


@pytest.fixture
def dummy_verification_result() -> VerificationResult:
    bundle = EvidenceBundle(
        claim="Dummy",
        passages=(),
        metadata=RetrievalMetadata(strategy_id="test", top_k=5),
    )
    return VerificationResult(
        label=VerificationLabel.SUPPORTS,
        confidence=0.9,
        evidence_bundle=bundle,
        metadata=VerificationMetadata(strategy_id="mock"),
    )


def test_failure_flag_immutable() -> None:
    flag = FailureFlag(code="A", description="B")
    with pytest.raises(ValidationError):
        flag.code = "C"

    # Must be hashable
    assert hash(flag)


def test_failure_analysis_result_immutable(
    dummy_verification_result: VerificationResult,
) -> None:
    flag = FailureFlag(code="A", description="B")
    result = FailureAnalysisResult(
        failure_flags=frozenset({flag}),
        severity=FailureSeverity.MEDIUM,
        verification_result=dummy_verification_result,
        metadata=FailureMetadata(strategy_id="mock"),
    )

    with pytest.raises(ValidationError):
        result.severity = FailureSeverity.HIGH


def test_protocol_compliance() -> None:
    assert issubclass(MockAnalyzer, BaseFailureAnalyzer)
    analyzer = MockAnalyzer()
    assert isinstance(analyzer, BaseFailureAnalyzer)


def test_compatibility_validation_fail_fast() -> None:
    analyzer = MockAnalyzer(reject=True)
    definition = DummyDefinition()

    with pytest.raises(FailureAnalysisConfigurationError, match="Incompatible"):
        analyzer.validate_compatibility(definition)


def test_orchestrator_delegation_and_equivalence(
    dummy_verification_result: VerificationResult,
) -> None:
    analyzer = MockAnalyzer()
    definition = DummyDefinition()
    orchestrator = FailureAnalyzer()

    result = orchestrator.analyze(
        "Claim", dummy_verification_result, definition, analyzer
    )

    assert analyzer.called_analyze
    assert result.severity == FailureSeverity.LOW
    assert len(result.failure_flags) == 1
    assert list(result.failure_flags)[0].code == "TEST_FLAG"
    # Identity preservation
    assert result.verification_result is dummy_verification_result


def test_orchestrator_exception_propagation(
    dummy_verification_result: VerificationResult,
) -> None:
    analyzer = MockAnalyzer(fail_analyze=True)
    definition = DummyDefinition()
    orchestrator = FailureAnalyzer()

    with pytest.raises(FailureAnalysisExecutionError, match="Runtime failure"):
        orchestrator.analyze("Claim", dummy_verification_result, definition, analyzer)
