"""Unit tests for the Evidence Retrieval subsystem framework."""

from collections.abc import Callable
from unittest.mock import MagicMock

import faiss
import numpy as np
import pytest
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from src.core.exceptions import RetrievalConfigurationError, RetrievalExecutionError
from src.core.retrieval.base import BaseRetriever, QueryEncoder
from src.core.retrieval.implementations import (
    BM25Retriever,
    FAISSRetriever,
    HybridRetriever,
)
from src.core.retrieval.retrieval_models import (
    BM25RetrievalDefinition,
    CorpusEntry,
    EvidenceBundle,
    EvidencePassage,
    FAISSRetrievalDefinition,
    HybridRetrievalDefinition,
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


@pytest.fixture
def dummy_corpus() -> tuple[CorpusEntry, ...]:
    return (
        CorpusEntry(document_id="doc1", span_id="1", text="the quick brown fox"),
        CorpusEntry(document_id="doc2", span_id="1", text="jumps over the lazy dog"),
        CorpusEntry(document_id="doc3", span_id="1", text="the quick brown dog"),
        CorpusEntry(document_id="doc4", span_id="1", text="foxes are fast"),
        CorpusEntry(document_id="doc5", span_id="1", text="lazy dogs are slow"),
    )


@pytest.fixture
def dummy_tokenizer() -> Callable[[str], list[str]]:
    return lambda text: text.lower().split()


@pytest.fixture
def bm25_index(
    dummy_corpus: tuple[CorpusEntry, ...],
    dummy_tokenizer: Callable[[str], list[str]],
) -> BM25Okapi:
    tokenized_corpus = [dummy_tokenizer(entry.text) for entry in dummy_corpus]
    return BM25Okapi(tokenized_corpus)


def test_corpus_entry_is_in_retrieval_models() -> None:
    """Verify CorpusEntry is importable from retrieval_models, not a BM25-specific module."""
    from src.core.retrieval.retrieval_models import CorpusEntry as CE

    assert CE is CorpusEntry


def test_corpus_entry_immutability() -> None:
    """Verify CorpusEntry is immutable."""
    entry = CorpusEntry(document_id="d1", span_id="s1", text="txt")
    with pytest.raises(Exception):
        entry.text = "mutated"


def test_bm25_retrieval_definition_immutability() -> None:
    """Test that BM25RetrievalDefinition is strictly immutable."""
    definition = BM25RetrievalDefinition(top_k=5)
    with pytest.raises(Exception):
        definition.top_k = 10


def test_bm25_retrieval_definition_requires_positive_top_k() -> None:
    """Verify top_k < 1 is rejected at construction."""
    with pytest.raises(Exception):
        BM25RetrievalDefinition(top_k=0)


def test_bm25_retrieval_definition_optional_threshold() -> None:
    """Verify score_threshold=None is valid."""
    definition = BM25RetrievalDefinition(top_k=5, score_threshold=None)
    assert definition.score_threshold is None


def test_bm25_retriever_satisfies_base_retriever_protocol(
    bm25_index: BM25Okapi,
    dummy_corpus: tuple[CorpusEntry, ...],
    dummy_tokenizer: Callable[[str], list[str]],
) -> None:
    """Verify isinstance(BM25Retriever(...), BaseRetriever)."""

    retriever = BM25Retriever(bm25_index, dummy_corpus, dummy_tokenizer)
    assert isinstance(retriever, BaseRetriever)


def test_bm25_retriever_accepts_bm25_definition(
    bm25_index: BM25Okapi,
    dummy_corpus: tuple[CorpusEntry, ...],
    dummy_tokenizer: Callable[[str], list[str]],
) -> None:
    """Verify validate_compatibility succeeds on valid definition."""

    retriever = BM25Retriever(bm25_index, dummy_corpus, dummy_tokenizer)
    definition = BM25RetrievalDefinition(top_k=3)
    retriever.validate_compatibility(definition)


def test_bm25_retriever_rejects_base_definition(
    bm25_index: BM25Okapi,
    dummy_corpus: tuple[CorpusEntry, ...],
    dummy_tokenizer: Callable[[str], list[str]],
) -> None:
    """Verify RetrievalConfigurationError on wrong definition type."""

    retriever = BM25Retriever(bm25_index, dummy_corpus, dummy_tokenizer)
    definition = RetrievalDefinition()
    with pytest.raises(RetrievalConfigurationError):
        retriever.validate_compatibility(definition)


def test_bm25_retriever_returns_top_k_passages(
    bm25_index: BM25Okapi,
    dummy_corpus: tuple[CorpusEntry, ...],
    dummy_tokenizer: Callable[[str], list[str]],
) -> None:
    """Verify exact count returned for a 5-entry corpus."""

    retriever = BM25Retriever(bm25_index, dummy_corpus, dummy_tokenizer)
    definition = BM25RetrievalDefinition(top_k=2)
    bundle = retriever.retrieve("quick brown fox", definition)
    assert len(bundle.passages) == 2


def test_bm25_retriever_descending_score_order(
    bm25_index: BM25Okapi,
    dummy_corpus: tuple[CorpusEntry, ...],
    dummy_tokenizer: Callable[[str], list[str]],
) -> None:
    """Verify passages are sorted by descending score."""

    retriever = BM25Retriever(bm25_index, dummy_corpus, dummy_tokenizer)
    definition = BM25RetrievalDefinition(top_k=5)
    bundle = retriever.retrieve("quick brown fox", definition)

    scores = [p.score for p in bundle.passages]
    assert scores == sorted(scores, reverse=True)


def test_bm25_retriever_score_threshold_filters_low_scores(
    bm25_index: BM25Okapi,
    dummy_corpus: tuple[CorpusEntry, ...],
    dummy_tokenizer: Callable[[str], list[str]],
) -> None:
    """Verify passages below threshold are excluded."""

    retriever = BM25Retriever(bm25_index, dummy_corpus, dummy_tokenizer)

    # First, get the scores without threshold
    unfiltered_bundle = retriever.retrieve("quick", BM25RetrievalDefinition(top_k=5))
    assert len(unfiltered_bundle.passages) > 0
    min_unfiltered_score = min([p.score for p in unfiltered_bundle.passages])

    # Now set threshold above the minimum
    threshold = min_unfiltered_score + 0.1
    definition = BM25RetrievalDefinition(top_k=5, score_threshold=threshold)
    filtered_bundle = retriever.retrieve("quick", definition)

    assert len(filtered_bundle.passages) < len(unfiltered_bundle.passages)
    assert all(p.score >= threshold for p in filtered_bundle.passages)


def test_bm25_retriever_score_threshold_none_returns_all(
    bm25_index: BM25Okapi,
    dummy_corpus: tuple[CorpusEntry, ...],
    dummy_tokenizer: Callable[[str], list[str]],
) -> None:
    """Verify None threshold disables filtering."""

    retriever = BM25Retriever(bm25_index, dummy_corpus, dummy_tokenizer)
    definition = BM25RetrievalDefinition(top_k=5, score_threshold=None)
    bundle = retriever.retrieve("quick", definition)
    assert len(bundle.passages) == 5  # Top k capped by corpus size


def test_bm25_retriever_identity(
    bm25_index: BM25Okapi,
    dummy_corpus: tuple[CorpusEntry, ...],
    dummy_tokenizer: Callable[[str], list[str]],
) -> None:
    """Verify document_id/span_id match corpus entry for top result."""

    retriever = BM25Retriever(bm25_index, dummy_corpus, dummy_tokenizer)
    definition = BM25RetrievalDefinition(top_k=1)
    # The claim exactly matches doc1 (id: doc1)
    bundle = retriever.retrieve("the quick brown fox", definition)
    assert bundle.passages[0].document_id == "doc1"
    assert bundle.passages[0].span_id == "1"
    assert bundle.passages[0].text == "the quick brown fox"


def test_bm25_retriever_returns_evidence_bundle(
    bm25_index: BM25Okapi,
    dummy_corpus: tuple[CorpusEntry, ...],
    dummy_tokenizer: Callable[[str], list[str]],
) -> None:
    """Verify return type is EvidenceBundle."""

    retriever = BM25Retriever(bm25_index, dummy_corpus, dummy_tokenizer)
    definition = BM25RetrievalDefinition(top_k=1)
    bundle = retriever.retrieve("fox", definition)
    assert isinstance(bundle, EvidenceBundle)


def test_bm25_retriever_metadata_strategy_id(
    bm25_index: BM25Okapi,
    dummy_corpus: tuple[CorpusEntry, ...],
    dummy_tokenizer: Callable[[str], list[str]],
) -> None:
    """Verify metadata.strategy_id == 'bm25'."""

    retriever = BM25Retriever(bm25_index, dummy_corpus, dummy_tokenizer)
    definition = BM25RetrievalDefinition(top_k=1)
    bundle = retriever.retrieve("fox", definition)
    assert bundle.metadata.strategy_id == "bm25"


def test_bm25_retriever_metadata_top_k(
    bm25_index: BM25Okapi,
    dummy_corpus: tuple[CorpusEntry, ...],
    dummy_tokenizer: Callable[[str], list[str]],
) -> None:
    """Verify metadata.top_k == definition.top_k."""

    retriever = BM25Retriever(bm25_index, dummy_corpus, dummy_tokenizer)
    definition = BM25RetrievalDefinition(top_k=3)
    bundle = retriever.retrieve("fox", definition)
    assert bundle.metadata.top_k == 3


def test_bm25_retriever_determinism(
    bm25_index: BM25Okapi,
    dummy_corpus: tuple[CorpusEntry, ...],
    dummy_tokenizer: Callable[[str], list[str]],
) -> None:
    """Verify identical claims produce identical bundles."""

    retriever = BM25Retriever(bm25_index, dummy_corpus, dummy_tokenizer)
    definition = BM25RetrievalDefinition(top_k=3)

    bundle1 = retriever.retrieve("quick brown dog", definition)
    bundle2 = retriever.retrieve("quick brown dog", definition)

    assert bundle1 == bundle2


def test_bm25_retriever_empty_corpus_returns_empty_bundle(
    dummy_tokenizer: Callable[[str], list[str]],
) -> None:
    """Empty corpus should return empty bundle without error."""

    mock_index = MagicMock()
    mock_index.get_scores.return_value = np.array([])

    retriever = BM25Retriever(mock_index, (), dummy_tokenizer)
    definition = BM25RetrievalDefinition(top_k=3)
    bundle = retriever.retrieve("fox", definition)
    assert len(bundle.passages) == 0


def test_bm25_retriever_propagates_execution_error(
    dummy_corpus: tuple[CorpusEntry, ...],
    dummy_tokenizer: Callable[[str], list[str]],
) -> None:
    """Mock get_scores to raise; verify RetrievalExecutionError."""

    class FaultyIndex(BM25Okapi):  # type: ignore[misc]
        def __init__(self) -> None:
            pass

        def get_scores(self, query: list[str]) -> list[float]:
            raise ValueError("Simulated fault")

    retriever = BM25Retriever(FaultyIndex(), dummy_corpus, dummy_tokenizer)
    definition = BM25RetrievalDefinition(top_k=3)

    with pytest.raises(
        RetrievalExecutionError, match="BM25 retrieval execution failed"
    ):
        retriever.retrieve("fox", definition)


def test_bm25_retriever_tokenizer_injection(
    bm25_index: BM25Okapi,
    dummy_corpus: tuple[CorpusEntry, ...],
) -> None:
    """Verify the injected tokenizer is called with the claim string during retrieval."""

    mock_tokenizer = MagicMock(return_value=["mocked"])
    retriever = BM25Retriever(bm25_index, dummy_corpus, mock_tokenizer)

    definition = BM25RetrievalDefinition(top_k=1)
    retriever.retrieve("fox", definition)

    mock_tokenizer.assert_called_once_with("fox")


def test_bm25_retriever_tokenizer_consistency(
    dummy_corpus: tuple[CorpusEntry, ...],
) -> None:
    """Verify that a custom tokenizer produces different top results than a whitespace tokenizer."""

    def whitespace_tokenizer(text: str) -> list[str]:
        return text.lower().split()

    # A dummy tokenizer that only ever extracts the word "dog", ignoring other words
    def dog_tokenizer(text: str) -> list[str]:
        return ["dog"] if "dog" in text.lower() else []

    whitespace_index = BM25Okapi([whitespace_tokenizer(e.text) for e in dummy_corpus])
    dog_index = BM25Okapi([dog_tokenizer(e.text) for e in dummy_corpus])

    retriever_whitespace = BM25Retriever(
        whitespace_index, dummy_corpus, whitespace_tokenizer
    )
    retriever_dog = BM25Retriever(dog_index, dummy_corpus, dog_tokenizer)

    definition = BM25RetrievalDefinition(top_k=1)

    # "lazy" query
    bundle1 = retriever_whitespace.retrieve("lazy", definition)
    # The dog tokenizer ignores "lazy", so it returns no matching scores, returning the first corpus entry by default stable sort
    bundle2 = retriever_dog.retrieve("lazy", definition)

    assert bundle1.passages[0].text != bundle2.passages[0].text


# ==============================================================================
# FAISS Retrieval Tests
# ==============================================================================


@pytest.fixture
def dummy_encoder() -> QueryEncoder:
    """A mock QueryEncoder that returns fixed deterministic vectors."""

    class MockEncoder:
        def encode(self, text: str) -> np.ndarray:
            # Deterministic mapping for tests based on text length
            val = float(len(text))
            # Return normalized 2D vector for 2-dim index
            vec = np.array([val, val + 1.0], dtype=np.float32)
            faiss.normalize_L2(vec.reshape(1, -1))
            return vec

    return MockEncoder()


@pytest.fixture
def dummy_faiss_index() -> faiss.Index:
    """A 2-dimensional IndexFlatIP populated with vectors corresponding to dummy_corpus."""
    index = faiss.IndexFlatIP(2)
    # 5 items in dummy_corpus
    vectors = []
    for i in range(5):
        val = float(i)
        vec = np.array([val, val + 1.0], dtype=np.float32)
        vectors.append(vec)

    matrix = np.vstack(vectors)
    faiss.normalize_L2(matrix)
    index.add(matrix)
    return index


def test_query_encoder_protocol_is_in_base() -> None:
    """Verify QueryEncoder is importable from base."""
    from src.core.retrieval.base import QueryEncoder as QE

    assert QE is QueryEncoder


def test_faiss_retriever_satisfies_base_retriever_protocol(
    dummy_faiss_index: faiss.Index,
    dummy_corpus: tuple[CorpusEntry, ...],
    dummy_encoder: QueryEncoder,
) -> None:
    """Verify isinstance(FAISSRetriever(...), BaseRetriever)."""

    retriever = FAISSRetriever(dummy_faiss_index, dummy_corpus, dummy_encoder)
    assert isinstance(retriever, BaseRetriever)


def test_faiss_retrieval_definition_immutability() -> None:
    """Test that FAISSRetrievalDefinition is strictly immutable."""
    definition = FAISSRetrievalDefinition(top_k=5)
    with pytest.raises(Exception):
        definition.top_k = 10


def test_faiss_retrieval_definition_requires_positive_top_k() -> None:
    """Verify top_k < 1 is rejected at construction."""
    with pytest.raises(Exception):
        FAISSRetrievalDefinition(top_k=0)


def test_faiss_retrieval_definition_optional_similarity_threshold() -> None:
    """Verify similarity_threshold=None is valid."""
    definition = FAISSRetrievalDefinition(top_k=5, similarity_threshold=None)
    assert definition.similarity_threshold is None


def test_faiss_retriever_accepts_faiss_definition(
    dummy_faiss_index: faiss.Index,
    dummy_corpus: tuple[CorpusEntry, ...],
    dummy_encoder: QueryEncoder,
) -> None:
    """Verify validate_compatibility succeeds on valid definition."""

    retriever = FAISSRetriever(dummy_faiss_index, dummy_corpus, dummy_encoder)
    definition = FAISSRetrievalDefinition(top_k=3)
    retriever.validate_compatibility(definition)


def test_faiss_retriever_rejects_base_definition(
    dummy_faiss_index: faiss.Index,
    dummy_corpus: tuple[CorpusEntry, ...],
    dummy_encoder: QueryEncoder,
) -> None:
    """Verify RetrievalConfigurationError on wrong definition type."""

    retriever = FAISSRetriever(dummy_faiss_index, dummy_corpus, dummy_encoder)
    definition = RetrievalDefinition()
    with pytest.raises(RetrievalConfigurationError):
        retriever.validate_compatibility(definition)


def test_faiss_retriever_rejects_bm25_definition(
    dummy_faiss_index: faiss.Index,
    dummy_corpus: tuple[CorpusEntry, ...],
    dummy_encoder: QueryEncoder,
) -> None:
    """Verify RetrievalConfigurationError on BM25 definition."""

    retriever = FAISSRetriever(dummy_faiss_index, dummy_corpus, dummy_encoder)
    definition = BM25RetrievalDefinition(top_k=3)
    with pytest.raises(RetrievalConfigurationError):
        retriever.validate_compatibility(definition)


def test_faiss_retriever_returns_top_k_passages(
    dummy_faiss_index: faiss.Index,
    dummy_corpus: tuple[CorpusEntry, ...],
    dummy_encoder: QueryEncoder,
) -> None:
    """Verify exact count returned."""

    retriever = FAISSRetriever(dummy_faiss_index, dummy_corpus, dummy_encoder)
    definition = FAISSRetrievalDefinition(top_k=2)
    bundle = retriever.retrieve("claim", definition)
    assert len(bundle.passages) == 2


def test_faiss_retriever_descending_score_order(
    dummy_faiss_index: faiss.Index,
    dummy_corpus: tuple[CorpusEntry, ...],
    dummy_encoder: QueryEncoder,
) -> None:
    """Verify passages are sorted by descending score."""

    retriever = FAISSRetriever(dummy_faiss_index, dummy_corpus, dummy_encoder)
    definition = FAISSRetrievalDefinition(top_k=5)
    bundle = retriever.retrieve("claim", definition)

    scores = [p.score for p in bundle.passages]
    assert scores == sorted(scores, reverse=True)


def test_faiss_retriever_similarity_threshold_filters_low_scores(
    dummy_faiss_index: faiss.Index,
    dummy_corpus: tuple[CorpusEntry, ...],
    dummy_encoder: QueryEncoder,
) -> None:
    """Verify passages below threshold are excluded."""

    retriever = FAISSRetriever(dummy_faiss_index, dummy_corpus, dummy_encoder)

    unfiltered_bundle = retriever.retrieve("claim", FAISSRetrievalDefinition(top_k=5))
    min_unfiltered_score = min([p.score for p in unfiltered_bundle.passages])

    threshold = min_unfiltered_score + 0.0001
    definition = FAISSRetrievalDefinition(top_k=5, similarity_threshold=threshold)
    filtered_bundle = retriever.retrieve("claim", definition)

    assert len(filtered_bundle.passages) < len(unfiltered_bundle.passages)
    assert all(p.score >= threshold for p in filtered_bundle.passages)


def test_faiss_retriever_similarity_threshold_none_returns_all(
    dummy_faiss_index: faiss.Index,
    dummy_corpus: tuple[CorpusEntry, ...],
    dummy_encoder: QueryEncoder,
) -> None:
    """Verify None threshold disables filtering."""

    retriever = FAISSRetriever(dummy_faiss_index, dummy_corpus, dummy_encoder)
    definition = FAISSRetrievalDefinition(top_k=5, similarity_threshold=None)
    bundle = retriever.retrieve("claim", definition)
    assert len(bundle.passages) == 5


def test_faiss_retriever_identity(
    dummy_faiss_index: faiss.Index,
    dummy_corpus: tuple[CorpusEntry, ...],
    dummy_encoder: QueryEncoder,
) -> None:
    """Verify document_id/span_id match corpus entry for top result."""

    retriever = FAISSRetriever(dummy_faiss_index, dummy_corpus, dummy_encoder)
    definition = FAISSRetrievalDefinition(top_k=1)

    bundle = retriever.retrieve("claim", definition)

    # Manually find which corpus entry got that score to verify identity
    # Our mock encoder returns a vector based on len("claim")
    vec = dummy_encoder.encode("claim")
    distances, indices = dummy_faiss_index.search(vec.reshape(1, -1), 1)
    expected_idx = indices[0][0]

    assert bundle.passages[0].document_id == dummy_corpus[expected_idx].document_id
    assert bundle.passages[0].span_id == dummy_corpus[expected_idx].span_id
    assert bundle.passages[0].text == dummy_corpus[expected_idx].text


def test_faiss_retriever_returns_evidence_bundle(
    dummy_faiss_index: faiss.Index,
    dummy_corpus: tuple[CorpusEntry, ...],
    dummy_encoder: QueryEncoder,
) -> None:
    """Verify return type is EvidenceBundle."""

    retriever = FAISSRetriever(dummy_faiss_index, dummy_corpus, dummy_encoder)
    definition = FAISSRetrievalDefinition(top_k=1)
    bundle = retriever.retrieve("claim", definition)
    assert isinstance(bundle, EvidenceBundle)


def test_faiss_retriever_metadata_strategy_id(
    dummy_faiss_index: faiss.Index,
    dummy_corpus: tuple[CorpusEntry, ...],
    dummy_encoder: QueryEncoder,
) -> None:
    """Verify metadata.strategy_id == 'faiss'."""

    retriever = FAISSRetriever(dummy_faiss_index, dummy_corpus, dummy_encoder)
    definition = FAISSRetrievalDefinition(top_k=1)
    bundle = retriever.retrieve("claim", definition)
    assert bundle.metadata.strategy_id == "faiss"


def test_faiss_retriever_metadata_top_k(
    dummy_faiss_index: faiss.Index,
    dummy_corpus: tuple[CorpusEntry, ...],
    dummy_encoder: QueryEncoder,
) -> None:
    """Verify metadata.top_k == definition.top_k."""

    retriever = FAISSRetriever(dummy_faiss_index, dummy_corpus, dummy_encoder)
    definition = FAISSRetrievalDefinition(top_k=3)
    bundle = retriever.retrieve("claim", definition)
    assert bundle.metadata.top_k == 3


def test_faiss_retriever_determinism(
    dummy_faiss_index: faiss.Index,
    dummy_corpus: tuple[CorpusEntry, ...],
    dummy_encoder: QueryEncoder,
) -> None:
    """Verify identical claims produce identical bundles."""

    retriever = FAISSRetriever(dummy_faiss_index, dummy_corpus, dummy_encoder)
    definition = FAISSRetrievalDefinition(top_k=3)

    bundle1 = retriever.retrieve("claim", definition)
    bundle2 = retriever.retrieve("claim", definition)

    assert bundle1 == bundle2


def test_faiss_retriever_handles_fewer_results_than_top_k(
    dummy_faiss_index: faiss.Index,
    dummy_corpus: tuple[CorpusEntry, ...],
    dummy_encoder: QueryEncoder,
) -> None:
    """Verify behavior when index has fewer elements than top_k."""

    retriever = FAISSRetriever(dummy_faiss_index, dummy_corpus, dummy_encoder)
    # We have 5 elements. Request 10.
    definition = FAISSRetrievalDefinition(top_k=10)
    bundle = retriever.retrieve("claim", definition)
    assert len(bundle.passages) == 5


def test_faiss_retriever_encoder_is_called_with_claim(
    dummy_faiss_index: faiss.Index,
    dummy_corpus: tuple[CorpusEntry, ...],
) -> None:
    """Verify encoder is invoked exactly once with the claim string."""

    mock_encoder = MagicMock(spec=QueryEncoder)
    mock_encoder.encode.return_value = np.array([1.0, 0.0], dtype=np.float32)

    retriever = FAISSRetriever(dummy_faiss_index, dummy_corpus, mock_encoder)
    definition = FAISSRetrievalDefinition(top_k=1)

    retriever.retrieve("claim", definition)
    mock_encoder.encode.assert_called_once_with("claim")


def test_faiss_retriever_encoder_exception_wraps_to_execution_error(
    dummy_faiss_index: faiss.Index,
    dummy_corpus: tuple[CorpusEntry, ...],
) -> None:
    """Verify encoder failures raise RetrievalExecutionError."""

    mock_encoder = MagicMock(spec=QueryEncoder)
    mock_encoder.encode.side_effect = ValueError("Encoder failed")

    retriever = FAISSRetriever(dummy_faiss_index, dummy_corpus, mock_encoder)
    definition = FAISSRetrievalDefinition(top_k=1)

    with pytest.raises(
        RetrievalExecutionError, match="FAISS retrieval execution failed"
    ):
        retriever.retrieve("claim", definition)


def test_faiss_retriever_search_exception_wraps_to_execution_error(
    dummy_corpus: tuple[CorpusEntry, ...],
    dummy_encoder: QueryEncoder,
) -> None:
    """Verify faiss.Index failures raise RetrievalExecutionError."""

    mock_index = MagicMock(spec=faiss.Index)
    mock_index.search.side_effect = RuntimeError("FAISS crashed")

    retriever = FAISSRetriever(mock_index, dummy_corpus, dummy_encoder)
    definition = FAISSRetrievalDefinition(top_k=1)

    with pytest.raises(
        RetrievalExecutionError, match="FAISS retrieval execution failed"
    ):
        retriever.retrieve("claim", definition)


# ==============================================================================
# Hybrid Retrieval Tests
# ==============================================================================


@pytest.fixture
def mock_bm25_retriever() -> MagicMock:
    return MagicMock(spec=BaseRetriever)


@pytest.fixture
def mock_faiss_retriever() -> MagicMock:
    return MagicMock(spec=BaseRetriever)


def test_hybrid_retrieval_definition_immutability() -> None:
    definition = HybridRetrievalDefinition(bm25_top_k=5, faiss_top_k=5, top_k=5)
    with pytest.raises(Exception):
        definition.top_k = 10


def test_hybrid_retrieval_definition_requires_positive_top_k() -> None:
    with pytest.raises(Exception):
        HybridRetrievalDefinition(bm25_top_k=5, faiss_top_k=5, top_k=0)


def test_hybrid_retrieval_definition_requires_positive_rrf_k() -> None:
    with pytest.raises(Exception):
        HybridRetrievalDefinition(bm25_top_k=5, faiss_top_k=5, top_k=5, rrf_k=0)


def test_hybrid_retrieval_definition_default_rrf_k_is_60() -> None:
    definition = HybridRetrievalDefinition(bm25_top_k=5, faiss_top_k=5, top_k=5)
    assert definition.rrf_k == 60


def test_hybrid_retriever_satisfies_base_retriever_protocol(
    mock_bm25_retriever: MagicMock,
    mock_faiss_retriever: MagicMock,
) -> None:
    retriever = HybridRetriever(mock_bm25_retriever, mock_faiss_retriever)
    assert isinstance(retriever, BaseRetriever)


def test_hybrid_retriever_accepts_hybrid_definition(
    mock_bm25_retriever: MagicMock,
    mock_faiss_retriever: MagicMock,
) -> None:
    retriever = HybridRetriever(mock_bm25_retriever, mock_faiss_retriever)
    definition = HybridRetrievalDefinition(bm25_top_k=5, faiss_top_k=5, top_k=5)
    retriever.validate_compatibility(definition)


def test_hybrid_retriever_rejects_base_definition(
    mock_bm25_retriever: MagicMock,
    mock_faiss_retriever: MagicMock,
) -> None:
    retriever = HybridRetriever(mock_bm25_retriever, mock_faiss_retriever)
    definition = RetrievalDefinition()
    with pytest.raises(RetrievalConfigurationError):
        retriever.validate_compatibility(definition)


def test_hybrid_retriever_rejects_bm25_definition(
    mock_bm25_retriever: MagicMock,
    mock_faiss_retriever: MagicMock,
) -> None:
    retriever = HybridRetriever(mock_bm25_retriever, mock_faiss_retriever)
    definition = BM25RetrievalDefinition(top_k=5)
    with pytest.raises(RetrievalConfigurationError):
        retriever.validate_compatibility(definition)


def test_hybrid_retriever_rejects_faiss_definition(
    mock_bm25_retriever: MagicMock,
    mock_faiss_retriever: MagicMock,
) -> None:
    retriever = HybridRetriever(mock_bm25_retriever, mock_faiss_retriever)
    definition = FAISSRetrievalDefinition(top_k=5)
    with pytest.raises(RetrievalConfigurationError):
        retriever.validate_compatibility(definition)


def test_hybrid_retriever_delegates_to_bm25_and_faiss(
    mock_bm25_retriever: MagicMock,
    mock_faiss_retriever: MagicMock,
) -> None:
    mock_bm25_retriever.retrieve.return_value = EvidenceBundle(
        claim="claim",
        passages=(),
        metadata=RetrievalMetadata(strategy_id="bm25", top_k=5),
    )
    mock_faiss_retriever.retrieve.return_value = EvidenceBundle(
        claim="claim",
        passages=(),
        metadata=RetrievalMetadata(strategy_id="faiss", top_k=5),
    )

    retriever = HybridRetriever(mock_bm25_retriever, mock_faiss_retriever)
    definition = HybridRetrievalDefinition(bm25_top_k=10, faiss_top_k=15, top_k=5)

    retriever.retrieve("claim", definition)

    # Assert BM25 called with correct ephemeral definition
    bm25_call_args = mock_bm25_retriever.retrieve.call_args[0]
    assert bm25_call_args[0] == "claim"
    assert isinstance(bm25_call_args[1], BM25RetrievalDefinition)
    assert bm25_call_args[1].top_k == 10

    # Assert FAISS called with correct ephemeral definition
    faiss_call_args = mock_faiss_retriever.retrieve.call_args[0]
    assert faiss_call_args[0] == "claim"
    assert isinstance(faiss_call_args[1], FAISSRetrievalDefinition)
    assert faiss_call_args[1].top_k == 15


def test_hybrid_retriever_propagates_bm25_execution_error(
    mock_bm25_retriever: MagicMock,
    mock_faiss_retriever: MagicMock,
) -> None:
    mock_bm25_retriever.retrieve.side_effect = RetrievalExecutionError("BM25 failed")
    retriever = HybridRetriever(mock_bm25_retriever, mock_faiss_retriever)
    definition = HybridRetrievalDefinition(bm25_top_k=5, faiss_top_k=5, top_k=5)

    with pytest.raises(RetrievalExecutionError, match="BM25 failed"):
        retriever.retrieve("claim", definition)


def test_hybrid_retriever_propagates_faiss_execution_error(
    mock_bm25_retriever: MagicMock,
    mock_faiss_retriever: MagicMock,
) -> None:
    mock_bm25_retriever.retrieve.return_value = EvidenceBundle(
        claim="claim",
        passages=(),
        metadata=RetrievalMetadata(strategy_id="bm25", top_k=5),
    )
    mock_faiss_retriever.retrieve.side_effect = RetrievalExecutionError("FAISS failed")
    retriever = HybridRetriever(mock_bm25_retriever, mock_faiss_retriever)
    definition = HybridRetrievalDefinition(bm25_top_k=5, faiss_top_k=5, top_k=5)

    with pytest.raises(RetrievalExecutionError, match="FAISS failed"):
        retriever.retrieve("claim", definition)


def test_hybrid_retriever_deduplicates_passages(
    mock_bm25_retriever: MagicMock,
    mock_faiss_retriever: MagicMock,
) -> None:
    p1 = EvidencePassage(document_id="doc1", span_id="span1", text="text1", score=1.0)
    p2 = EvidencePassage(document_id="doc2", span_id="span2", text="text2", score=0.9)

    # Both retrievers return p1. BM25 also returns p2.
    mock_bm25_retriever.retrieve.return_value = EvidenceBundle(
        claim="claim",
        passages=(p1, p2),
        metadata=RetrievalMetadata(strategy_id="bm25", top_k=2),
    )
    mock_faiss_retriever.retrieve.return_value = EvidenceBundle(
        claim="claim",
        passages=(p1,),
        metadata=RetrievalMetadata(strategy_id="faiss", top_k=1),
    )

    retriever = HybridRetriever(mock_bm25_retriever, mock_faiss_retriever)
    definition = HybridRetrievalDefinition(bm25_top_k=2, faiss_top_k=1, top_k=5)

    bundle = retriever.retrieve("claim", definition)

    # Should only have 2 passages total (p1 and p2)
    assert len(bundle.passages) == 2
    assert bundle.passages[0].document_id == "doc1"
    assert bundle.passages[1].document_id == "doc2"


def test_hybrid_retriever_returns_top_k_passages(
    mock_bm25_retriever: MagicMock,
    mock_faiss_retriever: MagicMock,
) -> None:
    passages_bm25 = tuple(
        EvidencePassage(document_id=f"doc_b_{i}", span_id="s", text="t", score=1.0)
        for i in range(5)
    )
    passages_faiss = tuple(
        EvidencePassage(document_id=f"doc_f_{i}", span_id="s", text="t", score=1.0)
        for i in range(5)
    )

    mock_bm25_retriever.retrieve.return_value = EvidenceBundle(
        claim="claim",
        passages=passages_bm25,
        metadata=RetrievalMetadata(strategy_id="bm25", top_k=5),
    )
    mock_faiss_retriever.retrieve.return_value = EvidenceBundle(
        claim="claim",
        passages=passages_faiss,
        metadata=RetrievalMetadata(strategy_id="faiss", top_k=5),
    )

    retriever = HybridRetriever(mock_bm25_retriever, mock_faiss_retriever)
    # 10 unique passages exist, but we only want top_k=3
    definition = HybridRetrievalDefinition(bm25_top_k=5, faiss_top_k=5, top_k=3)

    bundle = retriever.retrieve("claim", definition)

    assert len(bundle.passages) == 3


def test_hybrid_retriever_rrf_score_boosts_passages_in_both_lists(
    mock_bm25_retriever: MagicMock,
    mock_faiss_retriever: MagicMock,
) -> None:
    p_both = EvidencePassage(document_id="doc_both", span_id="s", text="t", score=1.0)
    p_bm25_only = EvidencePassage(
        document_id="doc_bm25", span_id="s", text="t", score=0.9
    )
    p_faiss_only = EvidencePassage(
        document_id="doc_faiss", span_id="s", text="t", score=0.8
    )

    # BM25 returns p_bm25_only at rank 1, p_both at rank 2
    mock_bm25_retriever.retrieve.return_value = EvidenceBundle(
        claim="claim",
        passages=(p_bm25_only, p_both),
        metadata=RetrievalMetadata(strategy_id="bm25", top_k=2),
    )
    # FAISS returns p_faiss_only at rank 1, p_both at rank 2
    mock_faiss_retriever.retrieve.return_value = EvidenceBundle(
        claim="claim",
        passages=(p_faiss_only, p_both),
        metadata=RetrievalMetadata(strategy_id="faiss", top_k=2),
    )

    retriever = HybridRetriever(mock_bm25_retriever, mock_faiss_retriever)
    definition = HybridRetrievalDefinition(
        bm25_top_k=2, faiss_top_k=2, top_k=5, rrf_k=60
    )

    bundle = retriever.retrieve("claim", definition)

    # p_both gets 1/(60+2) + 1/(60+2) = 2/62 = 0.0322
    # p_bm25_only gets 1/(60+1) = 1/61 = 0.0163
    # p_faiss_only gets 1/(60+1) = 1/61 = 0.0163
    # So p_both should be ranked first despite being rank 2 in both constituent lists

    assert bundle.passages[0].document_id == "doc_both"
    assert bundle.passages[1].score == 1.0 / 61.0
    assert bundle.passages[2].score == 1.0 / 61.0


def test_hybrid_retriever_descending_score_order(
    mock_bm25_retriever: MagicMock,
    mock_faiss_retriever: MagicMock,
) -> None:
    p1 = EvidencePassage(document_id="d1", span_id="s", text="t", score=1.0)
    p2 = EvidencePassage(document_id="d2", span_id="s", text="t", score=1.0)

    mock_bm25_retriever.retrieve.return_value = EvidenceBundle(
        claim="claim",
        passages=(p1, p2),
        metadata=RetrievalMetadata(strategy_id="bm25", top_k=2),
    )
    mock_faiss_retriever.retrieve.return_value = EvidenceBundle(
        claim="claim",
        passages=(p1,),
        metadata=RetrievalMetadata(strategy_id="faiss", top_k=1),
    )

    retriever = HybridRetriever(mock_bm25_retriever, mock_faiss_retriever)
    definition = HybridRetrievalDefinition(bm25_top_k=2, faiss_top_k=1, top_k=5)

    bundle = retriever.retrieve("claim", definition)

    scores = [p.score for p in bundle.passages]
    assert scores == sorted(scores, reverse=True)


def test_hybrid_retriever_tie_breaking_deterministic(
    mock_bm25_retriever: MagicMock,
    mock_faiss_retriever: MagicMock,
) -> None:
    # Both passages appear exactly once at the exact same rank in different lists
    # So they will have the exact same RRF score.
    p_b = EvidencePassage(document_id="doc_B", span_id="s", text="t", score=1.0)
    p_a = EvidencePassage(document_id="doc_A", span_id="s", text="t", score=1.0)

    mock_bm25_retriever.retrieve.return_value = EvidenceBundle(
        claim="claim",
        passages=(p_b,),
        metadata=RetrievalMetadata(strategy_id="bm25", top_k=1),
    )
    mock_faiss_retriever.retrieve.return_value = EvidenceBundle(
        claim="claim",
        passages=(p_a,),
        metadata=RetrievalMetadata(strategy_id="faiss", top_k=1),
    )

    retriever = HybridRetriever(mock_bm25_retriever, mock_faiss_retriever)
    definition = HybridRetrievalDefinition(bm25_top_k=1, faiss_top_k=1, top_k=2)

    bundle = retriever.retrieve("claim", definition)

    # Scores are equal. Tie break by document_id (lexicographical).
    # "doc_A" < "doc_B", so p_a should be first.
    assert bundle.passages[0].score == bundle.passages[1].score
    assert bundle.passages[0].document_id == "doc_A"
    assert bundle.passages[1].document_id == "doc_B"


def test_hybrid_retriever_no_overlap_union_result(
    mock_bm25_retriever: MagicMock,
    mock_faiss_retriever: MagicMock,
) -> None:
    p1 = EvidencePassage(document_id="d1", span_id="s", text="t", score=1.0)
    p2 = EvidencePassage(document_id="d2", span_id="s", text="t", score=1.0)

    mock_bm25_retriever.retrieve.return_value = EvidenceBundle(
        claim="claim",
        passages=(p1,),
        metadata=RetrievalMetadata(strategy_id="bm25", top_k=1),
    )
    mock_faiss_retriever.retrieve.return_value = EvidenceBundle(
        claim="claim",
        passages=(p2,),
        metadata=RetrievalMetadata(strategy_id="faiss", top_k=1),
    )

    retriever = HybridRetriever(mock_bm25_retriever, mock_faiss_retriever)
    definition = HybridRetrievalDefinition(bm25_top_k=1, faiss_top_k=1, top_k=5)

    bundle = retriever.retrieve("claim", definition)
    assert len(bundle.passages) == 2
    # RRF scores should be equal (rank 1), so tie-break kicks in: d1 < d2
    assert bundle.passages[0].document_id == "d1"
    assert bundle.passages[1].document_id == "d2"


def test_hybrid_retriever_full_overlap_deduplication(
    mock_bm25_retriever: MagicMock,
    mock_faiss_retriever: MagicMock,
) -> None:
    p1 = EvidencePassage(document_id="d1", span_id="s", text="t", score=1.0)
    p2 = EvidencePassage(document_id="d2", span_id="s", text="t", score=1.0)

    mock_bm25_retriever.retrieve.return_value = EvidenceBundle(
        claim="claim",
        passages=(p1, p2),
        metadata=RetrievalMetadata(strategy_id="bm25", top_k=2),
    )
    mock_faiss_retriever.retrieve.return_value = EvidenceBundle(
        claim="claim",
        passages=(p1, p2),
        metadata=RetrievalMetadata(strategy_id="faiss", top_k=2),
    )

    retriever = HybridRetriever(mock_bm25_retriever, mock_faiss_retriever)
    definition = HybridRetrievalDefinition(bm25_top_k=2, faiss_top_k=2, top_k=5)

    bundle = retriever.retrieve("claim", definition)
    # Output should have exactly 2 passages since both lists contain the exact same 2 passages
    assert len(bundle.passages) == 2


def test_hybrid_retriever_metadata_properties(
    mock_bm25_retriever: MagicMock,
    mock_faiss_retriever: MagicMock,
) -> None:
    mock_bm25_retriever.retrieve.return_value = EvidenceBundle(
        claim="claim",
        passages=(),
        metadata=RetrievalMetadata(strategy_id="bm25", top_k=1),
    )
    mock_faiss_retriever.retrieve.return_value = EvidenceBundle(
        claim="claim",
        passages=(),
        metadata=RetrievalMetadata(strategy_id="faiss", top_k=1),
    )

    retriever = HybridRetriever(mock_bm25_retriever, mock_faiss_retriever)
    definition = HybridRetrievalDefinition(bm25_top_k=1, faiss_top_k=1, top_k=3)

    bundle = retriever.retrieve("claim", definition)
    assert bundle.metadata.strategy_id == "hybrid"
    assert bundle.metadata.top_k == 3
