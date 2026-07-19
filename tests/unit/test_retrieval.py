"""Unit tests for the Evidence Retrieval subsystem framework."""

from unittest.mock import MagicMock

import pytest

from src.core.exceptions import RetrievalConfigurationError, RetrievalExecutionError
from src.core.retrieval.base import BaseRetriever
from src.core.retrieval.retrieval_models import (
    EvidenceBundle,
    EvidencePassage,
    RetrievalDefinition,
    RetrievalMetadata,
)
from src.core.retrieval.retriever import ClaimRetriever


@pytest.fixture
def dummy_definition() -> RetrievalDefinition:
    return RetrievalDefinition()


@pytest.fixture
def dummy_metadata() -> RetrievalMetadata:
    return RetrievalMetadata(strategy_id="test", top_k=5)


@pytest.fixture
def dummy_passage_1() -> EvidencePassage:
    return EvidencePassage(
        document_id="doc1",
        span_id="span1",
        text="text 1",
        score=0.9,
        metadata={"foo": "bar"},
    )


@pytest.fixture
def dummy_passage_2() -> EvidencePassage:
    return EvidencePassage(
        document_id="doc2",
        span_id="span2",
        text="text 2",
        score=0.8,
        metadata={},
    )


@pytest.fixture
def dummy_bundle(
    dummy_passage_1: EvidencePassage,
    dummy_passage_2: EvidencePassage,
    dummy_metadata: RetrievalMetadata,
) -> EvidenceBundle:
    return EvidenceBundle(
        claim="The claim.",
        passages=(dummy_passage_1, dummy_passage_2),
        metadata=dummy_metadata,
    )


def test_retrieval_definition_immutability() -> None:
    """Test that RetrievalDefinition is strictly immutable."""
    definition = RetrievalDefinition()
    with pytest.raises(Exception):
        definition.some_attr = "mutated"  # type: ignore[attr-defined]


def test_retrieval_metadata_immutability(dummy_metadata: RetrievalMetadata) -> None:
    """Test that RetrievalMetadata is strictly immutable."""
    with pytest.raises(Exception):
        dummy_metadata.strategy_id = "mutated"


def test_evidence_passage_immutability(dummy_passage_1: EvidencePassage) -> None:
    """Test that EvidencePassage is strictly immutable."""
    with pytest.raises(Exception):
        dummy_passage_1.text = "mutated"


def test_evidence_passage_identity(dummy_passage_1: EvidencePassage) -> None:
    """Test that document_id and span_id are independently addressable."""
    assert dummy_passage_1.document_id == "doc1"
    assert dummy_passage_1.span_id == "span1"


def test_evidence_bundle_immutability(dummy_bundle: EvidenceBundle) -> None:
    """Test that EvidenceBundle is strictly immutable."""
    with pytest.raises(Exception):
        dummy_bundle.claim = "mutated"


def test_evidence_bundle_ordering(
    dummy_passage_1: EvidencePassage,
    dummy_passage_2: EvidencePassage,
    dummy_bundle: EvidenceBundle,
) -> None:
    """Test that passages are ordered as provided."""
    # The fixture provides them in order 1, 2. We just verify the tuple preserves it.
    assert dummy_bundle.passages[0] is dummy_passage_1
    assert dummy_bundle.passages[1] is dummy_passage_2


def test_base_retriever_protocol_compliance() -> None:
    """Test that BaseRetriever is a runtime_checkable protocol."""

    class MockRetriever:
        def validate_compatibility(self, definition: RetrievalDefinition) -> None:
            pass

        def retrieve(
            self, claim: str, definition: RetrievalDefinition
        ) -> EvidenceBundle:
            # Return a dummy just to satisfy signature
            return MagicMock(spec=EvidenceBundle)

    assert isinstance(MockRetriever(), BaseRetriever)


def test_base_retriever_incompatible_definition() -> None:
    """Verify RetrievalConfigurationError propagates correctly."""
    mock_strategy = MagicMock(spec=BaseRetriever)
    mock_strategy.validate_compatibility.side_effect = RetrievalConfigurationError(
        "Incompatible"
    )

    with pytest.raises(RetrievalConfigurationError, match="Incompatible"):
        mock_strategy.validate_compatibility(RetrievalDefinition())


def test_claim_retriever_delegates_to_strategy(
    dummy_definition: RetrievalDefinition,
    dummy_bundle: EvidenceBundle,
) -> None:
    """Verify retrieve is called exactly once with the correct claim and definition."""
    retriever = ClaimRetriever()
    mock_strategy = MagicMock(spec=BaseRetriever)
    mock_strategy.retrieve.return_value = dummy_bundle
    claim = "Test claim"

    result = retriever.retrieve(claim, dummy_definition, mock_strategy)

    mock_strategy.retrieve.assert_called_once_with(claim, dummy_definition)
    assert result is dummy_bundle


def test_claim_retriever_no_compatibility_check(
    dummy_definition: RetrievalDefinition,
) -> None:
    """Verify validate_compatibility is never called by ClaimRetriever."""
    retriever = ClaimRetriever()
    mock_strategy = MagicMock(spec=BaseRetriever)

    retriever.retrieve("claim", dummy_definition, mock_strategy)

    mock_strategy.validate_compatibility.assert_not_called()


def test_claim_retriever_propagates_execution_error(
    dummy_definition: RetrievalDefinition,
) -> None:
    """Verify RetrievalExecutionError from strategy bubbles through."""
    retriever = ClaimRetriever()
    mock_strategy = MagicMock(spec=BaseRetriever)
    mock_strategy.retrieve.side_effect = RetrievalExecutionError("Failed")

    with pytest.raises(RetrievalExecutionError, match="Failed"):
        retriever.retrieve("claim", dummy_definition, mock_strategy)


def test_claim_retriever_returns_exact_bundle_identity(
    dummy_definition: RetrievalDefinition,
    dummy_bundle: EvidenceBundle,
) -> None:
    """Verify object identity is preserved through the orchestrator."""
    retriever = ClaimRetriever()
    mock_strategy = MagicMock(spec=BaseRetriever)
    mock_strategy.retrieve.return_value = dummy_bundle

    result = retriever.retrieve("claim", dummy_definition, mock_strategy)

    assert result is dummy_bundle
