"""Unit tests for the M8.5 Cross-Encoder Reranking subsystem."""

from collections.abc import Sequence

import pytest
from pydantic import ValidationError

from src.core.exceptions import RerankingConfigurationError, RerankingExecutionError
from src.core.reranking.base import CrossEncoderScorer
from src.core.reranking.implementations import CrossEncoderReranker
from src.core.reranking.reranking_models import RerankingDefinition
from src.core.retrieval.retrieval_models import (
    EvidenceBundle,
    EvidencePassage,
    RetrievalDefinition,
    RetrievalMetadata,
)


class MockScorer:
    def __init__(self, scores: list[float] | Exception):
        self.scores = scores
        self.called_with_query: str | None = None
        self.called_with_passages: Sequence[str] | None = None

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        self.called_with_query = query
        self.called_with_passages = passages
        if isinstance(self.scores, Exception):
            raise self.scores
        # Return a copy to avoid mutation side effects
        return list(self.scores)


def test_reranking_definition_immutable() -> None:
    definition = RerankingDefinition(top_k=10)
    with pytest.raises(ValidationError):
        definition.top_k = 5


def test_reranking_definition_validation() -> None:
    with pytest.raises(ValidationError):
        RerankingDefinition(top_k=0)

    with pytest.raises(ValidationError):
        RerankingDefinition(top_k=-1)

    definition = RerankingDefinition(top_k=5, score_threshold=0.5)
    assert definition.top_k == 5
    assert definition.score_threshold == 0.5


def test_scorer_protocol_compliance() -> None:
    scorer = MockScorer([1.0, 2.0])
    assert isinstance(scorer, CrossEncoderScorer)


def test_validate_compatibility() -> None:
    scorer = MockScorer([])
    reranker = CrossEncoderReranker(scorer)

    # Valid
    reranker.validate_compatibility(RerankingDefinition(top_k=5))

    # Invalid
    with pytest.raises(RerankingConfigurationError):
        reranker.validate_compatibility(RetrievalDefinition())  # type: ignore[arg-type]


def test_empty_bundle() -> None:
    scorer = MockScorer([1.0])  # Should not be called
    reranker = CrossEncoderReranker(scorer)
    definition = RerankingDefinition(top_k=5)

    empty_bundle = EvidenceBundle(
        claim="Claim",
        passages=(),
        metadata=RetrievalMetadata(strategy_id="bm25", top_k=10),
    )

    reranked = reranker.rerank(empty_bundle, definition)

    assert len(reranked.passages) == 0
    assert reranked.metadata.strategy_id == "cross_encoder"
    assert reranked.metadata.top_k == 5
    assert scorer.called_with_query is None


def test_score_propagation_and_metadata_preservation() -> None:
    scorer = MockScorer([0.9, 0.1])
    reranker = CrossEncoderReranker(scorer)
    definition = RerankingDefinition(top_k=5)

    bundle = EvidenceBundle(
        claim="Claim",
        passages=(
            EvidencePassage(
                document_id="doc1",
                span_id="1",
                text="text 1",
                score=10.0,
                metadata={"original_key": "val1"},
            ),
            EvidencePassage(
                document_id="doc2",
                span_id="2",
                text="text 2",
                score=5.0,
                metadata={"original_key": "val2", "retrieval_score": -1.0},
            ),
        ),
        metadata=RetrievalMetadata(strategy_id="bm25", top_k=10),
    )

    reranked = reranker.rerank(bundle, definition)

    assert len(reranked.passages) == 2

    p1 = reranked.passages[0]
    assert p1.document_id == "doc1"
    assert p1.span_id == "1"
    assert p1.text == "text 1"
    assert p1.score == 0.9
    assert p1.metadata["retrieval_score"] == 10.0
    assert p1.metadata["original_key"] == "val1"

    p2 = reranked.passages[1]
    assert p2.document_id == "doc2"
    assert p2.span_id == "2"
    assert p2.text == "text 2"
    assert p2.score == 0.1
    # Existing retrieval_score is correctly overwritten
    assert p2.metadata["retrieval_score"] == 5.0
    assert p2.metadata["original_key"] == "val2"


def test_ordering() -> None:
    # Passages in original order: 10.0, 5.0, 1.0
    # Cross-encoder scores: 0.1, 0.9, 0.5
    # Expected order: doc2 (0.9), doc3 (0.5), doc1 (0.1)
    scorer = MockScorer([0.1, 0.9, 0.5])
    reranker = CrossEncoderReranker(scorer)
    definition = RerankingDefinition(top_k=5)

    bundle = EvidenceBundle(
        claim="Claim",
        passages=(
            EvidencePassage(document_id="doc1", span_id="1", text="text 1", score=10.0),
            EvidencePassage(document_id="doc2", span_id="2", text="text 2", score=5.0),
            EvidencePassage(document_id="doc3", span_id="3", text="text 3", score=1.0),
        ),
        metadata=RetrievalMetadata(strategy_id="bm25", top_k=10),
    )

    reranked = reranker.rerank(bundle, definition)

    assert len(reranked.passages) == 3
    assert reranked.passages[0].document_id == "doc2"
    assert reranked.passages[1].document_id == "doc3"
    assert reranked.passages[2].document_id == "doc1"


def test_deterministic_tie_breaking() -> None:
    # Passages receive IDENTICAL cross encoder scores
    scorer = MockScorer([0.5, 0.5, 0.5])
    reranker = CrossEncoderReranker(scorer)
    definition = RerankingDefinition(top_k=5)

    bundle = EvidenceBundle(
        claim="Claim",
        passages=(
            EvidencePassage(document_id="B", span_id="2", text="t", score=1.0),
            EvidencePassage(document_id="A", span_id="2", text="t", score=2.0),
            EvidencePassage(document_id="A", span_id="1", text="t", score=3.0),
        ),
        metadata=RetrievalMetadata(strategy_id="bm25", top_k=10),
    )

    reranked = reranker.rerank(bundle, definition)

    # Expected order lexicographically by document_id, then span_id
    assert reranked.passages[0].document_id == "A"
    assert reranked.passages[0].span_id == "1"

    assert reranked.passages[1].document_id == "A"
    assert reranked.passages[1].span_id == "2"

    assert reranked.passages[2].document_id == "B"
    assert reranked.passages[2].span_id == "2"


def test_threshold_filtering() -> None:
    scorer = MockScorer([0.1, 0.9, 0.5])
    reranker = CrossEncoderReranker(scorer)
    definition = RerankingDefinition(top_k=5, score_threshold=0.6)

    bundle = EvidenceBundle(
        claim="Claim",
        passages=(
            EvidencePassage(document_id="doc1", span_id="1", text="text 1", score=10.0),
            EvidencePassage(document_id="doc2", span_id="2", text="text 2", score=5.0),
            EvidencePassage(document_id="doc3", span_id="3", text="text 3", score=1.0),
        ),
        metadata=RetrievalMetadata(strategy_id="bm25", top_k=10),
    )

    reranked = reranker.rerank(bundle, definition)

    assert len(reranked.passages) == 1
    assert reranked.passages[0].document_id == "doc2"
    assert reranked.passages[0].score == 0.9


def test_top_k_truncation() -> None:
    scorer = MockScorer([0.1, 0.9, 0.5])
    reranker = CrossEncoderReranker(scorer)
    definition = RerankingDefinition(top_k=2)

    bundle = EvidenceBundle(
        claim="Claim",
        passages=(
            EvidencePassage(document_id="doc1", span_id="1", text="text 1", score=10.0),
            EvidencePassage(document_id="doc2", span_id="2", text="text 2", score=5.0),
            EvidencePassage(document_id="doc3", span_id="3", text="text 3", score=1.0),
        ),
        metadata=RetrievalMetadata(strategy_id="bm25", top_k=10),
    )

    reranked = reranker.rerank(bundle, definition)

    assert len(reranked.passages) == 2
    assert reranked.passages[0].document_id == "doc2"
    assert reranked.passages[1].document_id == "doc3"


def test_scorer_length_mismatch() -> None:
    scorer = MockScorer([0.1])  # Only 1 score for 2 passages
    reranker = CrossEncoderReranker(scorer)
    definition = RerankingDefinition(top_k=5)

    bundle = EvidenceBundle(
        claim="Claim",
        passages=(
            EvidencePassage(document_id="doc1", span_id="1", text="text 1", score=10.0),
            EvidencePassage(document_id="doc2", span_id="2", text="text 2", score=5.0),
        ),
        metadata=RetrievalMetadata(strategy_id="bm25", top_k=10),
    )

    with pytest.raises(
        RerankingExecutionError, match="returned 1 scores for 2 passages"
    ):
        reranker.rerank(bundle, definition)


def test_scorer_exception_propagation() -> None:
    scorer = MockScorer(ValueError("Model OOM"))
    reranker = CrossEncoderReranker(scorer)
    definition = RerankingDefinition(top_k=5)

    bundle = EvidenceBundle(
        claim="Claim",
        passages=(
            EvidencePassage(document_id="doc1", span_id="1", text="text 1", score=10.0),
        ),
        metadata=RetrievalMetadata(strategy_id="bm25", top_k=10),
    )

    with pytest.raises(RerankingExecutionError, match="Model OOM"):
        reranker.rerank(bundle, definition)
