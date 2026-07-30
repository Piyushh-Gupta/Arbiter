"""Unit and integration tests for Hybrid Retrieval (C1.6)."""

import os
import shutil
import tempfile
import typing

import pytest

from src.core.exceptions import RetrievalConfigurationError, RetrievalExecutionError
from src.core.indexing.models import Chunk
from src.core.retrieval.bm25 import MetadataDocumentStore
from src.core.retrieval.hybrid import HybridRetriever
from src.core.retrieval.retrieval_models import (
    BM25RetrievalDefinition,
    DenseRetrievalDefinition,
    HybridRetrievalDefinition,
    RetrievalCandidate,
    RetrievalCandidateSet,
    RetrievalMetadata,
)


@pytest.fixture
def temp_dir() -> typing.Generator[str, None, None]:
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


@pytest.fixture
def dummy_metadata_path(temp_dir: str) -> str:
    path = os.path.join(temp_dir, "metadata.jsonl")
    chunks = [
        Chunk(
            span_id="zzz-chunk",
            document_id="doc1",
            text="first document in corpus",
            start_char=0,
            end_char=24,
            dataset_version="1.0",
            metadata={},
        ),
        Chunk(
            span_id="aaa-chunk",
            document_id="doc2",
            text="second document in corpus",
            start_char=0,
            end_char=25,
            dataset_version="1.0",
            metadata={},
        ),
        Chunk(
            span_id="mmm-chunk",
            document_id="doc3",
            text="third document in corpus",
            start_char=0,
            end_char=24,
            dataset_version="1.0",
            metadata={},
        ),
    ]
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c.model_dump_json() + "\n")
    return path


class DummyCandidateGenerator:
    """Mock candidate generator for deterministic testing of candidate fusion."""

    def __init__(self, candidates: tuple[RetrievalCandidate, ...]) -> None:
        self._candidates = candidates

    def generate_candidates(
        self, claim: str, definition: typing.Any
    ) -> RetrievalCandidateSet:
        return RetrievalCandidateSet(
            candidates=self._candidates,
            metadata=RetrievalMetadata(strategy_id="dummy", top_k=definition.top_k),
        )


def test_hybrid_definition_validation() -> None:
    # Requires at least one definition
    with pytest.raises(ValueError, match="at least one constituent definition"):
        HybridRetrievalDefinition(top_k=5)

    def_valid = HybridRetrievalDefinition(
        bm25_definition=BM25RetrievalDefinition(top_k=3),
        top_k=5,
        rrf_k=60,
    )
    assert def_valid.bm25_definition is not None
    assert def_valid.dense_definition is None


def test_hybrid_retriever_validation(dummy_metadata_path: str) -> None:
    store = MetadataDocumentStore(dummy_metadata_path)
    mock_bm25 = DummyCandidateGenerator(())

    with pytest.raises(ValueError, match="at least one candidate generator"):
        HybridRetriever(document_store=store)

    with pytest.raises(ValueError, match="requires a DocumentStore"):
        HybridRetriever(bm25_generator=mock_bm25, document_store=None)

    retriever = HybridRetriever(bm25_generator=mock_bm25, document_store=store)

    # Incompatible definition type
    with pytest.raises(RetrievalConfigurationError, match="HybridRetrievalDefinition"):
        retriever.validate_compatibility(BM25RetrievalDefinition(top_k=5))

    # Definition requests dense generator but none configured
    hybrid_def = HybridRetrievalDefinition(
        bm25_definition=BM25RetrievalDefinition(top_k=5),
        dense_definition=DenseRetrievalDefinition(top_k=5),
        top_k=5,
    )
    with pytest.raises(RetrievalConfigurationError, match="no DenseCandidateGenerator"):
        retriever.validate_compatibility(hybrid_def)


def test_rrf_fusion_mathematical_correctness(dummy_metadata_path: str) -> None:
    store = MetadataDocumentStore(dummy_metadata_path)

    # Lexical candidates: zzz-chunk (rank 1, score 10.0), mmm-chunk (rank 2, score 5.0)
    lex_cands = (
        RetrievalCandidate(
            span_id="zzz-chunk", score=10.0, metadata={"corpus_index": 0}
        ),
        RetrievalCandidate(
            span_id="mmm-chunk", score=5.0, metadata={"corpus_index": 2}
        ),
    )
    # Dense candidates: aaa-chunk (rank 1, score 0.9), mmm-chunk (rank 2, score 0.8)
    dense_cands = (
        RetrievalCandidate(
            span_id="aaa-chunk", score=0.9, metadata={"corpus_index": 1}
        ),
        RetrievalCandidate(
            span_id="mmm-chunk", score=0.8, metadata={"corpus_index": 2}
        ),
    )

    bm25_gen = DummyCandidateGenerator(lex_cands)
    dense_gen = DummyCandidateGenerator(dense_cands)

    retriever = HybridRetriever(
        bm25_generator=bm25_gen, dense_generator=dense_gen, document_store=store
    )

    definition = HybridRetrievalDefinition(
        bm25_definition=BM25RetrievalDefinition(top_k=2),
        dense_definition=DenseRetrievalDefinition(top_k=2),
        top_k=3,
        rrf_k=60,
    )

    bundle = retriever.retrieve("claim", definition)
    assert bundle.metadata.strategy_id == "hybrid"
    assert len(bundle.passages) == 3

    # mmm-chunk is present in both: lexical rank 2, dense rank 2
    # RRF score = 1/(60+2) + 1/(60+2) = 2/62 ≈ 0.032258
    # zzz-chunk: lexical rank 1, absent in dense => RRF score = 1/(60+1) ≈ 0.016393
    # aaa-chunk: dense rank 1, absent in lexical => RRF score = 1/(60+1) ≈ 0.016393

    top_passage = bundle.passages[0]
    assert top_passage.span_id == "mmm-chunk"
    assert top_passage.score == pytest.approx(2 / 62)
    assert top_passage.fusion_metadata is not None
    assert top_passage.fusion_metadata.lexical_rank == 2
    assert top_passage.fusion_metadata.lexical_score == 5.0
    assert top_passage.fusion_metadata.semantic_rank == 2
    assert top_passage.fusion_metadata.semantic_score == 0.80
    assert top_passage.fusion_metadata.rrf_score == pytest.approx(2 / 62)
    assert top_passage.fusion_metadata.retrieval_sources == ("bm25", "dense")


def test_corpus_order_tie_breaking(dummy_metadata_path: str) -> None:
    store = MetadataDocumentStore(dummy_metadata_path)

    # zzz-chunk has corpus_index=0, aaa-chunk has corpus_index=1
    # zzz-chunk is rank 1 in lexical (rrf = 1/61)
    # aaa-chunk is rank 1 in dense (rrf = 1/61)
    # Both have EXACTLY equal RRF score = 1/61.
    # Sorted by corpus_index ascending => zzz-chunk (0) comes BEFORE aaa-chunk (1),
    # even though 'aaa' comes before 'zzz' alphabetically!

    lex_cands = (
        RetrievalCandidate(
            span_id="zzz-chunk", score=10.0, metadata={"corpus_index": 0}
        ),
    )
    dense_cands = (
        RetrievalCandidate(
            span_id="aaa-chunk", score=0.9, metadata={"corpus_index": 1}
        ),
    )

    bm25_gen = DummyCandidateGenerator(lex_cands)
    dense_gen = DummyCandidateGenerator(dense_cands)

    retriever = HybridRetriever(
        bm25_generator=bm25_gen, dense_generator=dense_gen, document_store=store
    )
    definition = HybridRetrievalDefinition(
        bm25_definition=BM25RetrievalDefinition(top_k=1),
        dense_definition=DenseRetrievalDefinition(top_k=1),
        top_k=2,
        rrf_k=60,
    )

    bundle = retriever.retrieve("claim", definition)
    assert bundle.passages[0].span_id == "zzz-chunk"
    assert bundle.passages[1].span_id == "aaa-chunk"


def test_immutability_and_score_preservation(dummy_metadata_path: str) -> None:
    store = MetadataDocumentStore(dummy_metadata_path)

    orig_lex_cand = RetrievalCandidate(
        span_id="zzz-chunk", score=12.5, metadata={"corpus_index": 0}
    )
    bm25_gen = DummyCandidateGenerator((orig_lex_cand,))

    retriever = HybridRetriever(
        bm25_generator=bm25_gen, dense_generator=None, document_store=store
    )
    definition = HybridRetrievalDefinition(
        bm25_definition=BM25RetrievalDefinition(top_k=1),
        top_k=1,
        rrf_k=60,
    )

    bundle = retriever.retrieve("claim", definition)
    passage = bundle.passages[0]

    # Original candidate object is not mutated
    assert orig_lex_cand.score == 12.5
    # Provenance metadata retains original lexical score
    assert passage.fusion_metadata is not None
    assert passage.fusion_metadata.lexical_score == 12.5
    assert passage.fusion_metadata.semantic_score is None
    # Passage score is the computed RRF score (1/61)
    assert passage.score == pytest.approx(1 / 61)


def test_hybrid_retriever_exception_wrapping(dummy_metadata_path: str) -> None:
    class FailingCandidateGenerator:
        def generate_candidates(self, claim: str, definition: typing.Any) -> typing.Any:
            raise RuntimeError("Underlying vector search failed!")

    store = MetadataDocumentStore(dummy_metadata_path)
    retriever = HybridRetriever(
        bm25_generator=FailingCandidateGenerator(),
        dense_generator=None,
        document_store=store,
    )
    definition = HybridRetrievalDefinition(
        bm25_definition=BM25RetrievalDefinition(top_k=1),
        top_k=1,
    )

    with pytest.raises(
        RetrievalExecutionError, match="Hybrid retrieval execution failed"
    ):
        retriever.retrieve("claim", definition)
