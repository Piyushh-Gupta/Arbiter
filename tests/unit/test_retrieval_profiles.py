"""Unit tests for the M8.6 Retrieval Profiles subsystem."""

import pytest
from pydantic import ValidationError

from src.core.exceptions import (
    DuplicateRetrievalProfileError,
    RetrievalConfigurationError,
    RetrievalProfileNotFoundError,
)
from src.core.retrieval.retrieval_models import (
    EvidenceBundle,
    RetrievalDefinition,
    RetrievalMetadata,
    RetrievalProfile,
    RetrievalProfileRegistry,
)
from src.core.retrieval.retriever import ClaimRetriever


class MockDefinition(RetrievalDefinition):
    """Mock configuration for testing."""

    test_val: int = 1


class MockRetriever:
    """Mock retriever strategy for testing."""

    def __init__(self, reject: bool = False):
        self.reject = reject
        self.called_with_claim: str | None = None
        self.called_with_definition: RetrievalDefinition | None = None

    def validate_compatibility(self, definition: RetrievalDefinition) -> None:
        if self.reject:
            raise RetrievalConfigurationError("Incompatible configuration")

    def retrieve(self, claim: str, definition: RetrievalDefinition) -> EvidenceBundle:
        self.called_with_claim = claim
        self.called_with_definition = definition
        return EvidenceBundle(
            claim=claim,
            passages=(),
            metadata=RetrievalMetadata(strategy_id="mock", top_k=1),
        )


def test_retrieval_profile_immutable() -> None:
    definition = MockDefinition(test_val=1)
    strategy = MockRetriever()
    profile = RetrievalProfile(
        profile_id="test1", definition=definition, strategy=strategy
    )

    with pytest.raises(ValidationError):
        profile.profile_id = "test2"


def test_retrieval_profile_compatibility_validation() -> None:
    definition = MockDefinition(test_val=1)

    # Valid
    RetrievalProfile(
        profile_id="test1", definition=definition, strategy=MockRetriever(reject=False)
    )

    # Invalid
    with pytest.raises(RetrievalConfigurationError, match="Incompatible"):
        RetrievalProfile(
            profile_id="test1",
            definition=definition,
            strategy=MockRetriever(reject=True),
        )


def test_registry_immutable() -> None:
    profile = RetrievalProfile(
        profile_id="test1", definition=MockDefinition(), strategy=MockRetriever()
    )
    registry = RetrievalProfileRegistry(profiles=(profile,))

    with pytest.raises(ValidationError):
        registry.profiles = ()


def test_registry_empty_validation() -> None:
    # Empty registry should raise ValidationError at construction
    with pytest.raises(ValidationError, match="too_short"):
        RetrievalProfileRegistry(profiles=())


def test_registry_duplicate_detection() -> None:
    profile1 = RetrievalProfile(
        profile_id="test1", definition=MockDefinition(), strategy=MockRetriever()
    )
    profile2 = RetrievalProfile(
        profile_id="test1", definition=MockDefinition(), strategy=MockRetriever()
    )

    with pytest.raises(
        DuplicateRetrievalProfileError,
        match="Duplicate retrieval profile identifier: test1",
    ):
        RetrievalProfileRegistry(profiles=(profile1, profile2))


def test_registry_resolution_and_o1() -> None:
    profile1 = RetrievalProfile(
        profile_id="p1", definition=MockDefinition(), strategy=MockRetriever()
    )
    profile2 = RetrievalProfile(
        profile_id="p2", definition=MockDefinition(), strategy=MockRetriever()
    )
    registry = RetrievalProfileRegistry(profiles=(profile1, profile2))

    # O(1) resolution success
    resolved = registry.resolve("p2")
    assert resolved is profile2
    assert resolved.profile_id == "p2"

    # Not found
    with pytest.raises(
        RetrievalProfileNotFoundError, match="Retrieval profile not found: p3"
    ):
        registry.resolve("p3")


def test_execution_equivalence_and_identity() -> None:
    definition = MockDefinition(test_val=1)
    strategy = MockRetriever()
    profile = RetrievalProfile(
        profile_id="exec_test", definition=definition, strategy=strategy
    )
    registry = RetrievalProfileRegistry(profiles=(profile,))

    claim = "Test Claim"
    orchestrator = ClaimRetriever()

    # Direct execution
    bundle_direct = orchestrator.retrieve(claim, definition, strategy)

    # Resolve and execute via profile
    resolved_profile = registry.resolve("exec_test")
    bundle_resolved = orchestrator.retrieve(
        claim, resolved_profile.definition, resolved_profile.strategy
    )

    # Both should execute the same logic
    assert bundle_direct.claim == claim
    assert bundle_resolved.claim == claim

    # The exact same definition and strategy instances are passed
    assert resolved_profile.definition is definition
    assert resolved_profile.strategy is strategy

    # Strategy should have been called with identical objects
    assert strategy.called_with_claim == claim
    assert strategy.called_with_definition is definition
