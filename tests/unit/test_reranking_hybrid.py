"""Comprehensive unit tests for C1.7 Cross-Encoder Reranking."""

import os
import shutil
import tempfile
import typing
from collections.abc import Sequence

import pytest
from pydantic import ValidationError

from src.core.exceptions import (
    DuplicateRerankingProfileError,
    RerankingExecutionError,
    RerankingProfileNotFoundError,
)
from src.core.indexing.models import Chunk
from src.core.reranking.base import BaseCrossEncoderScorer
from src.core.reranking.implementations import CrossEncoderReranker
from src.core.reranking.reranking_models import (
    CrossEncoderModelMetadata,
    RerankingDefinition,
    RerankingProfile,
    RerankingProfileRegistry,
)
from src.core.retrieval.bm25 import MetadataDocumentStore
from src.core.retrieval.retrieval_models import (
    FusionMetadata,
    RetrievalCandidate,
    RetrievalCandidateSet,
    RetrievalMetadata,
)
from src.core.verification.implementations import NLIVerifier
from src.core.verification.verification_models import NLIVerificationDefinition


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
            span_id="span-1",
            document_id="doc1",
            text="First document passage",
            start_char=0,
            end_char=22,
            dataset_version="1.0",
            metadata={},
        ),
        Chunk(
            span_id="span-2",
            document_id="doc2",
            text="Second document passage",
            start_char=0,
            end_char=23,
            dataset_version="1.0",
            metadata={},
        ),
        Chunk(
            span_id="span-3",
            document_id="doc3",
            text="Third document passage",
            start_char=0,
            end_char=22,
            dataset_version="1.0",
            metadata={},
        ),
    ]
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c.model_dump_json() + "\n")
    return path


class MockBatchScorer(BaseCrossEncoderScorer):
    """Mock scorer recording batch calls."""

    def __init__(self, score_map: dict[str, float] | Exception) -> None:
        self._score_map = score_map
        self.batch_calls: list[list[str]] = []

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        self.batch_calls.append(list(passages))
        if isinstance(self._score_map, Exception):
            raise self._score_map
        return [self._score_map.get(p, 0.0) for p in passages]


def test_cross_encoder_model_metadata() -> None:
    meta = CrossEncoderModelMetadata(
        model_identifier="cross-encoder/ms-marco-MiniLM-L-6-v2",
        tokenizer_identifier="cross-encoder/ms-marco-MiniLM-L-6-v2",
        inference_framework="sentence-transformers",
        execution_device="cpu",
        max_sequence_length=512,
    )
    assert meta.model_identifier == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert meta.max_sequence_length == 512


def test_reranking_definition_bounds_validation() -> None:
    # top_k_output cannot exceed top_k_input
    with pytest.raises(ValidationError, match="top_k_output cannot exceed top_k_input"):
        RerankingDefinition(top_k_input=2, top_k_output=5)

    def_valid = RerankingDefinition(top_k_input=10, top_k_output=5, batch_size=4)
    assert def_valid.top_k_input == 10
    assert def_valid.top_k_output == 5
    assert def_valid.top_k == 5
    assert def_valid.batch_size == 4


def test_batch_inference_slicing(dummy_metadata_path: str) -> None:
    store = MetadataDocumentStore(dummy_metadata_path)
    score_map = {
        "First document passage": 0.2,
        "Second document passage": 0.9,
        "Third document passage": 0.5,
    }
    scorer = MockBatchScorer(score_map)
    reranker = CrossEncoderReranker(scorer=scorer, document_store=store)

    cands = (
        RetrievalCandidate(span_id="span-1", score=0.01, metadata={"corpus_index": 0}),
        RetrievalCandidate(span_id="span-2", score=0.02, metadata={"corpus_index": 1}),
        RetrievalCandidate(span_id="span-3", score=0.03, metadata={"corpus_index": 2}),
    )
    cand_set = RetrievalCandidateSet(
        candidates=cands,
        metadata=RetrievalMetadata(strategy_id="hybrid", top_k=3),
    )

    # Force batch size = 2 to verify contiguous batch slicing
    definition = RerankingDefinition(top_k_input=3, top_k_output=2, batch_size=2)

    bundle = reranker.rerank("claim", cand_set, definition)
    assert len(bundle.passages) == 2

    # Verify scorer received 2 batches: batch 1 with 2 items, batch 2 with 1 item
    assert len(scorer.batch_calls) == 2
    assert len(scorer.batch_calls[0]) == 2
    assert len(scorer.batch_calls[1]) == 1

    # Second passage ("Second document passage", score 0.9) comes first
    assert bundle.passages[0].span_id == "span-2"
    assert bundle.passages[0].score == 0.9
    assert bundle.passages[1].span_id == "span-3"
    assert bundle.passages[1].score == 0.5


def test_score_preservation(dummy_metadata_path: str) -> None:
    store = MetadataDocumentStore(dummy_metadata_path)
    scorer = MockBatchScorer({"First document passage": 0.95})
    reranker = CrossEncoderReranker(scorer=scorer, document_store=store)

    fusion_meta = FusionMetadata(
        lexical_rank=1,
        lexical_score=14.2,
        semantic_rank=2,
        semantic_score=0.88,
        rrf_score=0.032,
        retrieval_sources=("bm25", "dense"),
    )
    cand = RetrievalCandidate(
        span_id="span-1",
        score=0.032,
        metadata={"corpus_index": 0},
        fusion_metadata=fusion_meta,
    )
    cand_set = RetrievalCandidateSet(
        candidates=(cand,),
        metadata=RetrievalMetadata(strategy_id="hybrid", top_k=1),
    )

    definition = RerankingDefinition(top_k_input=1, top_k_output=1)
    bundle = reranker.rerank("claim", cand_set, definition)

    passage = bundle.passages[0]
    # Primary score is updated to Cross-Encoder score
    assert passage.score == 0.95

    # Prior fusion evidence is preserved untouched
    assert passage.fusion_metadata is not None
    assert passage.fusion_metadata.lexical_score == 14.2
    assert passage.fusion_metadata.semantic_score == 0.88
    assert passage.fusion_metadata.rrf_score == 0.032

    # Rerank metadata records stage-2 details
    assert passage.rerank_metadata is not None
    assert passage.rerank_metadata.rerank_score == 0.95
    assert passage.rerank_metadata.rerank_rank == 1
    assert passage.rerank_metadata.prior_fusion_metadata == fusion_meta


def test_corpus_order_tie_breaking_for_reranker(dummy_metadata_path: str) -> None:
    store = MetadataDocumentStore(dummy_metadata_path)
    # Both passages receive mathematically identical Cross-Encoder score = 0.8
    scorer = MockBatchScorer(
        {
            "First document passage": 0.8,
            "Second document passage": 0.8,
        }
    )
    reranker = CrossEncoderReranker(scorer=scorer, document_store=store)

    cands = (
        RetrievalCandidate(span_id="span-1", score=0.5, metadata={"corpus_index": 0}),
        RetrievalCandidate(span_id="span-2", score=0.6, metadata={"corpus_index": 1}),
    )
    cand_set = RetrievalCandidateSet(
        candidates=cands,
        metadata=RetrievalMetadata(strategy_id="hybrid", top_k=2),
    )

    definition = RerankingDefinition(top_k_input=2, top_k_output=2)
    bundle = reranker.rerank("claim", cand_set, definition)

    # Both scores equal (0.8) => tie-breaker places span-1 (corpus_index=0) before span-2 (corpus_index=1)
    assert bundle.passages[0].span_id == "span-1"
    assert bundle.passages[1].span_id == "span-2"


def test_reranker_exception_wrapping(dummy_metadata_path: str) -> None:
    store = MetadataDocumentStore(dummy_metadata_path)
    failing_scorer = MockBatchScorer(RuntimeError("Model CUDA OOM"))
    reranker = CrossEncoderReranker(scorer=failing_scorer, document_store=store)

    cand = RetrievalCandidate(span_id="span-1", score=0.5, metadata={"corpus_index": 0})
    cand_set = RetrievalCandidateSet(
        candidates=(cand,),
        metadata=RetrievalMetadata(strategy_id="hybrid", top_k=1),
    )
    definition = RerankingDefinition(top_k_input=1, top_k_output=1)

    with pytest.raises(
        RerankingExecutionError, match="Cross-encoder batch inference failed"
    ):
        reranker.rerank("claim", cand_set, definition)


def test_reranking_profile_registry() -> None:
    scorer = MockBatchScorer({})
    reranker = CrossEncoderReranker(scorer=scorer)
    definition = RerankingDefinition(top_k_input=5, top_k_output=2)

    profile = RerankingProfile(
        profile_id="rr_profile",
        definition=definition,
        strategy=reranker,
    )
    registry = RerankingProfileRegistry(profiles=(profile,))

    resolved = registry.resolve("rr_profile")
    assert resolved is profile

    with pytest.raises(DuplicateRerankingProfileError):
        RerankingProfileRegistry(profiles=(profile, profile))

    with pytest.raises(RerankingProfileNotFoundError):
        registry.resolve("non_existent")


def test_end_to_end_reranking_to_verification_pipeline(
    dummy_metadata_path: str,
) -> None:
    store = MetadataDocumentStore(dummy_metadata_path)
    scorer = MockBatchScorer(
        {
            "First document passage": 0.9,
            "Second document passage": 0.1,
        }
    )
    reranker = CrossEncoderReranker(scorer=scorer, document_store=store)
    definition = RerankingDefinition(top_k_input=2, top_k_output=1)

    cands = (
        RetrievalCandidate(span_id="span-1", score=0.5, metadata={"corpus_index": 0}),
        RetrievalCandidate(span_id="span-2", score=0.6, metadata={"corpus_index": 1}),
    )
    cand_set = RetrievalCandidateSet(
        candidates=cands,
        metadata=RetrievalMetadata(strategy_id="hybrid", top_k=2),
    )

    # 1. Rerank candidates into EvidenceBundle
    bundle = reranker.rerank("Claim text", cand_set, definition)
    assert len(bundle.passages) == 1
    assert bundle.passages[0].span_id == "span-1"

    # 2. Feed EvidenceBundle downstream into NLIVerifier
    from src.core.bootstrap import DummyNLIModel

    verifier = NLIVerifier(model=DummyNLIModel(), strategy_id="dummy_nli")
    ver_def = NLIVerificationDefinition(top_k=1)

    ver_result = verifier.verify("Claim text", bundle, ver_def)
    assert ver_result.label.name == "SUPPORTS"
    assert ver_result.evidence_bundle is bundle
