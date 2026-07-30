"""Unit tests for the Evidence Retrieval subsystem framework."""

from unittest.mock import MagicMock

import faiss
import numpy as np
import pytest

from src.core.exceptions import RetrievalConfigurationError, RetrievalExecutionError
from src.core.retrieval.base import BaseRetriever, QueryEncoder
from src.core.retrieval.implementations import DenseRetriever
from src.core.retrieval.retrieval_models import (
    BM25RetrievalDefinition,
    CorpusEntry,
    DenseRetrievalDefinition,
    EvidenceBundle,
    EvidencePassage,
    HybridRetrievalDefinition,
    RetrievalDefinition,
    RetrievalMetadata,
)
from src.core.retrieval.retriever import ClaimRetriever


class DummyRetriever(BaseRetriever):
    def __init__(self, bundle: EvidenceBundle) -> None:
        self.bundle = bundle

    def validate_compatibility(self, definition: RetrievalDefinition) -> None:
        pass

    def retrieve(self, claim: str, definition: RetrievalDefinition) -> EvidenceBundle:
        return self.bundle


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
        document_id="doc2", span_id="span2", text="text 2", score=0.8, metadata={}
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
    dummy_definition: RetrievalDefinition, dummy_bundle: EvidenceBundle
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
    dummy_definition: RetrievalDefinition, dummy_bundle: EvidenceBundle
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


def test_corpus_entry_is_in_retrieval_models() -> None:
    """Verify CorpusEntry is importable from retrieval_models, not a BM25-specific module."""
    from src.core.retrieval.retrieval_models import CorpusEntry as CE

    assert CE is CorpusEntry


def test_corpus_entry_immutability() -> None:
    """Verify CorpusEntry is immutable."""
    entry = CorpusEntry(document_id="d1", span_id="s1", text="txt")
    with pytest.raises(Exception):
        entry.text = "mutated"


@pytest.fixture
def dummy_encoder() -> QueryEncoder:
    """A mock QueryEncoder that returns fixed deterministic vectors."""

    class MockEncoder:
        @property
        def model_id(self) -> str:
            return "mock-encoder-v1"

        @property
        def embedding_dimension(self) -> int:
            return 2

        @property
        def device(self) -> str:
            return "cpu"

        @property
        def pooling_strategy(self) -> str:
            return "mean"

        @property
        def normalization_strategy(self) -> str:
            return "l2"

        @property
        def model_revision(self) -> str | None:
            return None

        def is_ready(self) -> bool:
            return True

        def encode(self, text: str) -> np.ndarray:
            val = float(len(text))
            vec = np.array([val, val + 1.0], dtype=np.float32)
            faiss.normalize_L2(vec.reshape(1, -1))
            return vec

    return MockEncoder()


@pytest.fixture
def dummy_faiss_index() -> faiss.Index:
    """A 2-dimensional IndexFlatIP populated with vectors corresponding to dummy_corpus."""
    index = faiss.IndexFlatIP(2)
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
    """Verify isinstance(DenseRetriever(...), BaseRetriever)."""
    retriever = DenseRetriever(dummy_faiss_index, dummy_corpus, dummy_encoder)
    assert isinstance(retriever, BaseRetriever)


def test_faiss_retrieval_definition_immutability() -> None:
    """Test that DenseRetrievalDefinition is strictly immutable."""
    definition = DenseRetrievalDefinition(top_k=5)
    with pytest.raises(Exception):
        definition.top_k = 10


def test_faiss_retrieval_definition_requires_positive_top_k() -> None:
    """Verify top_k < 1 is rejected at construction."""
    with pytest.raises(Exception):
        DenseRetrievalDefinition(top_k=0)


def test_faiss_retrieval_definition_optional_min_score() -> None:
    """Verify min_score=None is valid."""
    definition = DenseRetrievalDefinition(top_k=5, min_score=None)
    assert definition.min_score is None


def test_faiss_retriever_accepts_faiss_definition(
    dummy_faiss_index: faiss.Index,
    dummy_corpus: tuple[CorpusEntry, ...],
    dummy_encoder: QueryEncoder,
) -> None:
    """Verify validate_compatibility succeeds on valid definition."""
    retriever = DenseRetriever(dummy_faiss_index, dummy_corpus, dummy_encoder)
    definition = DenseRetrievalDefinition(top_k=3)
    retriever.validate_compatibility(definition)


def test_faiss_retriever_rejects_base_definition(
    dummy_faiss_index: faiss.Index,
    dummy_corpus: tuple[CorpusEntry, ...],
    dummy_encoder: QueryEncoder,
) -> None:
    """Verify RetrievalConfigurationError on wrong definition type."""
    retriever = DenseRetriever(dummy_faiss_index, dummy_corpus, dummy_encoder)
    definition = RetrievalDefinition()
    with pytest.raises(RetrievalConfigurationError):
        retriever.validate_compatibility(definition)


def test_faiss_retriever_returns_top_k_passages(
    dummy_faiss_index: faiss.Index,
    dummy_corpus: tuple[CorpusEntry, ...],
    dummy_encoder: QueryEncoder,
) -> None:
    """Verify exact count returned."""
    retriever = DenseRetriever(dummy_faiss_index, dummy_corpus, dummy_encoder)
    definition = DenseRetrievalDefinition(top_k=2)
    bundle = retriever.retrieve("claim", definition)
    assert len(bundle.passages) == 2


def test_faiss_retriever_descending_score_order(
    dummy_faiss_index: faiss.Index,
    dummy_corpus: tuple[CorpusEntry, ...],
    dummy_encoder: QueryEncoder,
) -> None:
    """Verify passages are sorted by descending score."""
    retriever = DenseRetriever(dummy_faiss_index, dummy_corpus, dummy_encoder)
    definition = DenseRetrievalDefinition(top_k=5)
    bundle = retriever.retrieve("claim", definition)
    scores = [p.score for p in bundle.passages]
    assert scores == sorted(scores, reverse=True)


def test_faiss_retriever_min_score_filters_low_scores(
    dummy_faiss_index: faiss.Index,
    dummy_corpus: tuple[CorpusEntry, ...],
    dummy_encoder: QueryEncoder,
) -> None:
    """Verify passages below threshold are excluded."""
    retriever = DenseRetriever(dummy_faiss_index, dummy_corpus, dummy_encoder)
    unfiltered_bundle = retriever.retrieve("claim", DenseRetrievalDefinition(top_k=5))
    min_unfiltered_score = min([p.score for p in unfiltered_bundle.passages])
    threshold = min_unfiltered_score + 0.0001
    definition = DenseRetrievalDefinition(top_k=5, min_score=threshold)
    filtered_bundle = retriever.retrieve("claim", definition)
    assert len(filtered_bundle.passages) < len(unfiltered_bundle.passages)
    assert all((p.score >= threshold for p in filtered_bundle.passages))


def test_faiss_retriever_min_score_none_returns_all(
    dummy_faiss_index: faiss.Index,
    dummy_corpus: tuple[CorpusEntry, ...],
    dummy_encoder: QueryEncoder,
) -> None:
    """Verify None threshold disables filtering."""
    retriever = DenseRetriever(dummy_faiss_index, dummy_corpus, dummy_encoder)
    definition = DenseRetrievalDefinition(top_k=5, min_score=None)
    bundle = retriever.retrieve("claim", definition)
    assert len(bundle.passages) == 5


def test_faiss_retriever_identity(
    dummy_faiss_index: faiss.Index,
    dummy_corpus: tuple[CorpusEntry, ...],
    dummy_encoder: QueryEncoder,
) -> None:
    """Verify document_id/span_id match corpus entry for top result."""
    retriever = DenseRetriever(dummy_faiss_index, dummy_corpus, dummy_encoder)
    definition = DenseRetrievalDefinition(top_k=1)
    bundle = retriever.retrieve("claim", definition)
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
    retriever = DenseRetriever(dummy_faiss_index, dummy_corpus, dummy_encoder)
    definition = DenseRetrievalDefinition(top_k=1)
    bundle = retriever.retrieve("claim", definition)
    assert isinstance(bundle, EvidenceBundle)


def test_faiss_retriever_metadata_strategy_id(
    dummy_faiss_index: faiss.Index,
    dummy_corpus: tuple[CorpusEntry, ...],
    dummy_encoder: QueryEncoder,
) -> None:
    """Verify metadata.strategy_id == 'faiss'."""
    retriever = DenseRetriever(dummy_faiss_index, dummy_corpus, dummy_encoder)
    definition = DenseRetrievalDefinition(top_k=1)
    bundle = retriever.retrieve("claim", definition)
    assert bundle.metadata.strategy_id == "faiss"


def test_faiss_retriever_metadata_top_k(
    dummy_faiss_index: faiss.Index,
    dummy_corpus: tuple[CorpusEntry, ...],
    dummy_encoder: QueryEncoder,
) -> None:
    """Verify metadata.top_k == definition.top_k."""
    retriever = DenseRetriever(dummy_faiss_index, dummy_corpus, dummy_encoder)
    definition = DenseRetrievalDefinition(top_k=3)
    bundle = retriever.retrieve("claim", definition)
    assert bundle.metadata.top_k == 3


def test_faiss_retriever_determinism(
    dummy_faiss_index: faiss.Index,
    dummy_corpus: tuple[CorpusEntry, ...],
    dummy_encoder: QueryEncoder,
) -> None:
    """Verify identical claims produce identical bundles."""
    retriever = DenseRetriever(dummy_faiss_index, dummy_corpus, dummy_encoder)
    definition = DenseRetrievalDefinition(top_k=3)
    bundle1 = retriever.retrieve("claim", definition)
    bundle2 = retriever.retrieve("claim", definition)
    assert bundle1 == bundle2


def test_faiss_retriever_handles_fewer_results_than_top_k(
    dummy_faiss_index: faiss.Index,
    dummy_corpus: tuple[CorpusEntry, ...],
    dummy_encoder: QueryEncoder,
) -> None:
    """Verify behavior when index has fewer elements than top_k."""
    retriever = DenseRetriever(dummy_faiss_index, dummy_corpus, dummy_encoder)
    definition = DenseRetrievalDefinition(top_k=10)
    bundle = retriever.retrieve("claim", definition)
    assert len(bundle.passages) == 5


def test_faiss_retriever_encoder_is_called_with_claim(
    dummy_faiss_index: faiss.Index, dummy_corpus: tuple[CorpusEntry, ...]
) -> None:
    """Verify encoder is invoked exactly once with the claim string."""
    mock_encoder = MagicMock(spec=QueryEncoder)
    mock_encoder.encode.return_value = np.array([1.0, 0.0], dtype=np.float32)
    retriever = DenseRetriever(dummy_faiss_index, dummy_corpus, mock_encoder)
    definition = DenseRetrievalDefinition(top_k=1)
    retriever.retrieve("claim", definition)
    mock_encoder.encode.assert_called_once_with("claim")


def test_faiss_retriever_encoder_exception_wraps_to_execution_error(
    dummy_faiss_index: faiss.Index, dummy_corpus: tuple[CorpusEntry, ...]
) -> None:
    """Verify encoder failures raise RetrievalExecutionError."""
    mock_encoder = MagicMock(spec=QueryEncoder)
    mock_encoder.encode.side_effect = ValueError("Encoder failed")
    retriever = DenseRetriever(dummy_faiss_index, dummy_corpus, mock_encoder)
    definition = DenseRetrievalDefinition(top_k=1)
    with pytest.raises(
        RetrievalExecutionError, match="FAISS retrieval execution failed"
    ):
        retriever.retrieve("claim", definition)


def test_faiss_retriever_search_exception_wraps_to_execution_error(
    dummy_corpus: tuple[CorpusEntry, ...], dummy_encoder: QueryEncoder
) -> None:
    """Verify faiss.Index failures raise RetrievalExecutionError."""
    mock_index = MagicMock(spec=faiss.Index)
    mock_index.search.side_effect = RuntimeError("FAISS crashed")
    retriever = DenseRetriever(mock_index, dummy_corpus, dummy_encoder)
    definition = DenseRetrievalDefinition(top_k=1)
    with pytest.raises(
        RetrievalExecutionError, match="FAISS retrieval execution failed"
    ):
        retriever.retrieve("claim", definition)


@pytest.fixture
def mock_bm25_retriever() -> MagicMock:
    return MagicMock(spec=BaseRetriever)


@pytest.fixture
def mock_faiss_retriever() -> MagicMock:
    return MagicMock(spec=BaseRetriever)


def test_hybrid_retrieval_definition_immutability() -> None:
    bm25_def = BM25RetrievalDefinition(top_k=5)
    dense_def = DenseRetrievalDefinition(top_k=5)
    definition = HybridRetrievalDefinition(
        bm25_definition=bm25_def, dense_definition=dense_def, top_k=5
    )
    with pytest.raises(Exception):
        definition.top_k = 10


def test_hybrid_retrieval_definition_requires_positive_top_k() -> None:
    bm25_def = BM25RetrievalDefinition(top_k=5)
    dense_def = DenseRetrievalDefinition(top_k=5)
    with pytest.raises(Exception):
        HybridRetrievalDefinition(
            bm25_definition=bm25_def, dense_definition=dense_def, top_k=0
        )


def test_hybrid_retrieval_definition_requires_positive_rrf_k() -> None:
    bm25_def = BM25RetrievalDefinition(top_k=5)
    dense_def = DenseRetrievalDefinition(top_k=5)
    with pytest.raises(Exception):
        HybridRetrievalDefinition(
            bm25_definition=bm25_def, dense_definition=dense_def, top_k=5, rrf_k=0
        )


def test_hybrid_retrieval_definition_default_rrf_k_is_60() -> None:
    bm25_def = BM25RetrievalDefinition(top_k=5)
    dense_def = DenseRetrievalDefinition(top_k=5)
    definition = HybridRetrievalDefinition(
        bm25_definition=bm25_def, dense_definition=dense_def, top_k=5
    )
    assert definition.rrf_k == 60
