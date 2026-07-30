"""Unit tests for the M9.1 Verification Framework."""

import pytest
from pydantic import ValidationError

from src.core.exceptions import (
    VerificationConfigurationError,
    VerificationExecutionError,
)
from src.core.retrieval.retrieval_models import EvidenceBundle, RetrievalMetadata
from src.core.verification.base import BaseVerifier
from src.core.verification.verification_models import (
    PassageVerificationResult,
    VerificationDefinition,
    VerificationLabel,
    VerificationMetadata,
    VerificationResult,
)
from src.core.verification.verifier import ClaimVerifier


@pytest.fixture
def dummy_bundle() -> EvidenceBundle:
    return EvidenceBundle(
        claim="Dummy claim",
        passages=(),
        metadata=RetrievalMetadata(strategy_id="test", top_k=5),
    )


def test_verification_label() -> None:
    assert VerificationLabel.SUPPORTS == "SUPPORTS"
    assert VerificationLabel.REFUTES == "REFUTES"
    assert VerificationLabel.NOT_ENOUGH_INFO == "NOT_ENOUGH_INFO"

    with pytest.raises(ValueError):
        VerificationLabel("INVALID_LABEL")


def test_verification_metadata_immutable() -> None:
    metadata = VerificationMetadata(strategy_id="test_strategy")
    assert metadata.strategy_id == "test_strategy"

    with pytest.raises(ValidationError):
        metadata.strategy_id = "other"


def test_verification_definition_immutable() -> None:
    definition = VerificationDefinition()

    with pytest.raises(ValidationError):
        definition.new_field = "value"  # type: ignore[attr-defined]


def test_verification_result_immutable_and_validation(
    dummy_bundle: EvidenceBundle,
) -> None:
    metadata = VerificationMetadata(strategy_id="test")

    # Valid construction with confidence
    result = VerificationResult(
        label=VerificationLabel.SUPPORTS,
        confidence=0.95,
        evidence_bundle=dummy_bundle,
        metadata=metadata,
    )
    assert result.label == VerificationLabel.SUPPORTS
    assert result.confidence == 0.95
    assert result.evidence_bundle is dummy_bundle
    assert result.metadata is metadata

    # Valid construction without confidence
    result_no_conf = VerificationResult(
        label=VerificationLabel.REFUTES,
        confidence=None,
        evidence_bundle=dummy_bundle,
        metadata=metadata,
    )
    assert result_no_conf.confidence is None

    # Immutable check
    with pytest.raises(ValidationError):
        result.confidence = 0.5

    # Confidence validation (must be between 0.0 and 1.0)
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        VerificationResult(
            label=VerificationLabel.SUPPORTS,
            confidence=-0.1,
            evidence_bundle=dummy_bundle,
            metadata=metadata,
        )

    with pytest.raises(ValidationError, match="less than or equal to 1"):
        VerificationResult(
            label=VerificationLabel.SUPPORTS,
            confidence=1.1,
            evidence_bundle=dummy_bundle,
            metadata=metadata,
        )


class MockVerifier:
    """Mock verifier to test protocol compliance and orchestrator delegation."""

    def __init__(self, reject: bool = False, fail_exec: bool = False):
        self.reject = reject
        self.fail_exec = fail_exec
        self.called_with_claim: str | None = None
        self.called_with_bundle: EvidenceBundle | None = None
        self.called_with_definition: VerificationDefinition | None = None

    def validate_compatibility(self, definition: VerificationDefinition) -> None:
        if self.reject:
            raise VerificationConfigurationError("Incompatible definition")

    def verify_passages(
        self, claim: str, bundle: EvidenceBundle
    ) -> tuple[PassageVerificationResult, ...]:
        return ()

    def verify(
        self,
        claim: str,
        bundle: EvidenceBundle,
        definition: VerificationDefinition,
    ) -> VerificationResult:
        if self.fail_exec:
            raise VerificationExecutionError("Runtime failure")
        self.called_with_claim = claim
        self.called_with_bundle = bundle
        self.called_with_definition = definition

        return VerificationResult(
            label=VerificationLabel.SUPPORTS,
            confidence=0.8,
            evidence_bundle=bundle,
            metadata=VerificationMetadata(strategy_id="mock"),
        )


def test_base_verifier_protocol() -> None:
    assert isinstance(MockVerifier(), BaseVerifier)


def test_orchestrator_delegation_and_equivalence(dummy_bundle: EvidenceBundle) -> None:
    verifier = MockVerifier()
    orchestrator = ClaimVerifier()
    definition = VerificationDefinition()
    claim = "Test claim"

    # Direct execution
    result_direct = verifier.verify(claim, dummy_bundle, definition)

    # Orchestrator execution
    verifier.called_with_claim = None  # reset
    result_orch = orchestrator.verify(claim, dummy_bundle, definition, verifier)

    # Execution equivalence
    assert result_direct.label == result_orch.label
    assert result_direct.confidence == result_orch.confidence

    # Object identity preservation
    assert verifier.called_with_claim == claim
    assert verifier.called_with_bundle is dummy_bundle
    assert verifier.called_with_definition is definition
    assert result_orch.evidence_bundle is dummy_bundle


def test_exception_propagation(dummy_bundle: EvidenceBundle) -> None:
    verifier = MockVerifier(fail_exec=True)
    orchestrator = ClaimVerifier()
    definition = VerificationDefinition()

    with pytest.raises(VerificationExecutionError, match="Runtime failure"):
        orchestrator.verify("Test", dummy_bundle, definition, verifier)


def test_validate_compatibility_fail_fast() -> None:
    verifier = MockVerifier(reject=True)
    definition = VerificationDefinition()

    with pytest.raises(VerificationConfigurationError, match="Incompatible"):
        verifier.validate_compatibility(definition)
