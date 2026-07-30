"""Comprehensive unit tests for C1.10 Production Retrieval Optimization."""

import os
import shutil
import tempfile
import typing

import pytest
from pydantic import ValidationError
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from src.core.bootstrap import DummyCrossEncoderScorer, build_optimization_registry
from src.core.cache import (
    CacheDefinition,
    CachedRetriever,
    CacheKeyGenerator,
    InMemoryRetrievalCache,
)
from src.core.config import Settings
from src.core.exceptions import (
    DuplicateOptimizationProfileError,
    OptimizationProfileNotFoundError,
)
from src.core.indexing.models import Chunk
from src.core.reranking.implementations import CrossEncoderReranker
from src.core.reranking.reranking_models import RerankingDefinition
from src.core.retrieval.bm25 import (
    BM25CandidateGenerator,
    MetadataDocumentStore,
    WhitespaceTokenizer,
)
from src.core.retrieval.hybrid import HybridRetriever
from src.core.retrieval.optimization import (
    BoundedSemaphoreConcurrencyLimiter,
    ExecutionPolicy,
    OptimizationController,
    OptimizationDefinition,
    OptimizationProfile,
    OptimizationProfileRegistry,
    TelemetryCollector,
)
from src.core.retrieval.retrieval_models import (
    BM25RetrievalDefinition,
    EvidenceBundle,
    EvidencePassage,
    HybridRetrievalDefinition,
    RetrievalMetadata,
)


@pytest.fixture
def temp_dir() -> typing.Generator[str, None, None]:
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


@pytest.fixture
def sample_bundle() -> EvidenceBundle:
    return EvidenceBundle(
        claim="Sample claim text",
        passages=(
            EvidencePassage(
                document_id="doc1",
                span_id="span-1",
                text="Sample passage text 1",
                score=0.9,
            ),
        ),
        metadata=RetrievalMetadata(strategy_id="test", top_k=1),
    )


class DummyRetriever:
    def __init__(self, bundle: EvidenceBundle) -> None:
        self.bundle = bundle

    def retrieve(self, claim: str) -> EvidenceBundle:
        return self.bundle


def test_execution_policy_validation() -> None:
    with pytest.raises(ValidationError):
        ExecutionPolicy(retrieval_batch_size=0)

    with pytest.raises(ValidationError):
        ExecutionPolicy(max_concurrent_requests=-1)

    policy = ExecutionPolicy(
        retrieval_batch_size=32,
        reranking_batch_size=16,
        max_concurrent_requests=8,
        request_timeout_ms=3000.0,
    )
    assert policy.retrieval_batch_size == 32
    assert policy.max_concurrent_requests == 8


def test_concurrency_limiter_behavior() -> None:
    limiter = BoundedSemaphoreConcurrencyLimiter(max_concurrency=2)
    assert limiter.max_concurrency == 2

    assert limiter.acquire(timeout_ms=100.0) is True
    assert limiter.acquire(timeout_ms=100.0) is True

    # Third acquire times out
    assert limiter.acquire(timeout_ms=50.0) is False

    limiter.release()
    assert limiter.acquire(timeout_ms=100.0) is True

    limiter.release()
    limiter.release()


def test_telemetry_collector_aggregation(sample_bundle: EvidenceBundle) -> None:
    collector = TelemetryCollector()
    dummy_ret = DummyRetriever(sample_bundle)
    policy = ExecutionPolicy(request_timeout_ms=1000.0)
    limiter = BoundedSemaphoreConcurrencyLimiter(max_concurrency=4)
    controller = OptimizationController(
        execution_policy=policy,
        concurrency_limiter=limiter,
        telemetry_collector=collector,
    )

    # Execute 3 times
    for _ in range(3):
        bundle, metrics = controller.execute("claim text", dummy_ret)
        assert bundle == sample_bundle
        assert metrics.total_latency_ms >= 0.0

    snap = collector.snapshot()
    assert snap.total_requests == 3
    assert snap.average_latency_ms >= 0.0
    assert snap.throughput_qps >= 0.0


def test_optimization_profile_registry() -> None:
    policy = ExecutionPolicy()
    limiter = BoundedSemaphoreConcurrencyLimiter(max_concurrency=2)
    definition = OptimizationDefinition(execution_policy=policy)
    profile = OptimizationProfile(
        profile_id="test_profile",
        definition=definition,
        execution_policy=policy,
        concurrency_limiter=limiter,
    )

    registry = OptimizationProfileRegistry(profiles=(profile,))
    assert registry.resolve("test_profile") is profile

    with pytest.raises(DuplicateOptimizationProfileError):
        OptimizationProfileRegistry(profiles=(profile, profile))

    with pytest.raises(OptimizationProfileNotFoundError):
        registry.resolve("non_existent")


def test_bootstrap_optimization_registry() -> None:
    config = Settings()
    registry = build_optimization_registry(config)
    resolved = registry.resolve("default_optimization")
    assert resolved.execution_policy.max_concurrent_requests == 4


def test_end_to_end_optimized_retrieval_determinism(temp_dir: str) -> None:
    # 1. Setup DocumentStore & Corpus
    path = os.path.join(temp_dir, "metadata.jsonl")
    chunk1 = Chunk(
        span_id="span-1",
        document_id="doc1",
        text="Oxygen is necessary for combustion.",
        start_char=0,
        end_char=35,
        dataset_version="1.0",
        metadata={"corpus_index": 0},
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(chunk1.model_dump_json() + "\n")

    doc_store = MetadataDocumentStore(path)
    bm25_index = BM25Okapi([WhitespaceTokenizer().tokenize(chunk1.text)])
    bm25_gen = BM25CandidateGenerator(
        index=bm25_index,
        span_ids=["span-1"],
        tokenizer=WhitespaceTokenizer(),
    )
    hybrid = HybridRetriever(
        bm25_generator=bm25_gen,
        dense_generator=None,
        document_store=doc_store,
    )
    reranker = CrossEncoderReranker(
        scorer=DummyCrossEncoderScorer(),
        document_store=doc_store,
    )

    c_def = CacheDefinition(enabled=True)
    cache = InMemoryRetrievalCache(c_def)
    key_gen = CacheKeyGenerator()

    cached_retriever = CachedRetriever(
        retriever=hybrid,
        cache=cache,
        key_generator=key_gen,
        definition=c_def,
        reranker=reranker,
        corpus_version="v1.0",
    )

    ret_def = HybridRetrievalDefinition(
        bm25_definition=BM25RetrievalDefinition(top_k=1), top_k=1
    )
    rr_def = RerankingDefinition(top_k_input=1, top_k_output=1)

    # Direct Unoptimized Invocation
    direct_bundle = cached_retriever.retrieve(
        claim="Oxygen role in combustion",
        retrieval_definition=ret_def,
        reranking_definition=rr_def,
    )

    # Optimized Invocation via OptimizationController
    policy = ExecutionPolicy(request_timeout_ms=5000.0)
    limiter = BoundedSemaphoreConcurrencyLimiter(max_concurrency=4)
    telemetry = TelemetryCollector()
    controller = OptimizationController(
        execution_policy=policy,
        concurrency_limiter=limiter,
        telemetry_collector=telemetry,
    )

    opt_bundle, metrics = controller.execute(
        claim="Oxygen role in combustion",
        retriever=cached_retriever,
        definition=ret_def,
        reranker=None,
        reranking_definition=rr_def,
    )

    # Determinism Assertion: Outputs are identical
    assert direct_bundle == opt_bundle
    assert metrics.passage_count == len(direct_bundle.passages)
    assert telemetry.snapshot().total_requests == 1
