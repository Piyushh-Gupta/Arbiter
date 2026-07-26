"""Unit tests for the M9.3 Verification Profiles framework."""

import pytest
from pydantic import ValidationError

from src.core.exceptions import (
    DuplicateVerificationProfileError,
    VerificationConfigurationError,
    VerificationProfileNotFoundError,
)
from src.core.retrieval.retrieval_models import EvidenceBundle, RetrievalMetadata
from src.core.verification.base import BaseVerifier
from src.core.verification.verification_models import (
    VerificationDefinition,
    VerificationLabel,
    VerificationMetadata,
    VerificationProfile,
    VerificationProfileRegistry,
    VerificationResult,
)
from src.core.verification.verifier import ClaimVerifier


class DummyDefinition(VerificationDefinition):
    pass


class MockVerifier(BaseVerifier):
    def __init__(self, reject: bool = False):
        self.reject = reject
        self.called_verify = False

    def validate_compatibility(self, definition: VerificationDefinition) -> None:
        if self.reject:
            raise VerificationConfigurationError("Incompatible definition")

    def verify(
        self,
        claim: str,
        bundle: EvidenceBundle,
        definition: VerificationDefinition,
    ) -> VerificationResult:
        self.called_verify = True
        return VerificationResult(
            label=VerificationLabel.SUPPORTS,
            confidence=0.9,
            evidence_bundle=bundle,
            metadata=VerificationMetadata(strategy_id="mock"),
        )


@pytest.fixture
def dummy_bundle() -> EvidenceBundle:
    return EvidenceBundle(
        claim="Dummy",
        passages=(),
        metadata=RetrievalMetadata(strategy_id="test", top_k=5),
    )


def test_verification_profile_immutable_and_validates() -> None:
    definition = DummyDefinition()
    verifier = MockVerifier()

    profile = VerificationProfile(
        profile_id="test_profile",
        definition=definition,
        verifier=verifier,
    )

    assert profile.profile_id == "test_profile"
    assert profile.definition is definition
    assert profile.verifier is verifier

    # Immutability
    with pytest.raises(ValidationError):
        profile.profile_id = "other"


def test_verification_profile_fail_fast_compatibility() -> None:
    definition = DummyDefinition()
    verifier = MockVerifier(reject=True)

    with pytest.raises(VerificationConfigurationError, match="Incompatible"):
        VerificationProfile(
            profile_id="test_profile",
            definition=definition,
            verifier=verifier,
        )


def test_verification_profile_registry_immutable_and_resolves() -> None:
    profile1 = VerificationProfile(
        profile_id="p1",
        definition=DummyDefinition(),
        verifier=MockVerifier(),
    )
    profile2 = VerificationProfile(
        profile_id="p2",
        definition=DummyDefinition(),
        verifier=MockVerifier(),
    )

    registry = VerificationProfileRegistry(profiles=(profile1, profile2))

    # O(1) resolution
    resolved = registry.resolve("p2")
    assert resolved is profile2

    # Immutability
    with pytest.raises(ValidationError):
        registry.profiles = ()


def test_registry_min_length_validation() -> None:
    with pytest.raises(ValidationError):
        VerificationProfileRegistry(profiles=())


def test_registry_duplicate_detection() -> None:
    profile1 = VerificationProfile(
        profile_id="dup",
        definition=DummyDefinition(),
        verifier=MockVerifier(),
    )
    profile2 = VerificationProfile(
        profile_id="dup",
        definition=DummyDefinition(),
        verifier=MockVerifier(),
    )

    with pytest.raises(
        DuplicateVerificationProfileError,
        match="Duplicate verification profile identifier: dup",
    ):
        VerificationProfileRegistry(profiles=(profile1, profile2))


def test_registry_not_found() -> None:
    profile = VerificationProfile(
        profile_id="p1",
        definition=DummyDefinition(),
        verifier=MockVerifier(),
    )
    registry = VerificationProfileRegistry(profiles=(profile,))

    with pytest.raises(
        VerificationProfileNotFoundError,
        match="Verification profile not found: missing",
    ):
        registry.resolve("missing")


def test_execution_equivalence(dummy_bundle: EvidenceBundle) -> None:
    verifier = MockVerifier()
    definition = DummyDefinition()
    profile = VerificationProfile(
        profile_id="p1",
        definition=definition,
        verifier=verifier,
    )
    registry = VerificationProfileRegistry(profiles=(profile,))

    # Orchestrator is profile-agnostic
    orchestrator = ClaimVerifier()
    resolved = registry.resolve("p1")

    assert resolved.definition is definition
    assert resolved.verifier is verifier

    result = orchestrator.verify(
        "Claim", dummy_bundle, resolved.definition, resolved.verifier
    )

    assert verifier.called_verify
    assert result.label == VerificationLabel.SUPPORTS
    assert result.evidence_bundle is dummy_bundle
