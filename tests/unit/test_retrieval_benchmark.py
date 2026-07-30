"""Comprehensive unit tests for C1.9 Retrieval Benchmarking & Evaluation."""

import os
import shutil
import tempfile
import typing

import pytest
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from src.core.bootstrap import DummyCrossEncoderScorer, build_benchmark_registry
from src.core.cache import (
    CacheDefinition,
    CachedRetriever,
    CacheKeyGenerator,
    InMemoryRetrievalCache,
)
from src.core.config import Settings
from src.core.exceptions import (
    BenchmarkProfileNotFoundError,
    DuplicateBenchmarkProfileError,
    DuplicateMetricError,
    MetricNotFoundError,
)
from src.core.indexing.models import Chunk
from src.core.reranking.implementations import CrossEncoderReranker
from src.core.reranking.reranking_models import RerankingDefinition
from src.core.retrieval.benchmarking import (
    BenchmarkDataset,
    BenchmarkDefinition,
    BenchmarkEnvironmentMetadata,
    BenchmarkProfile,
    BenchmarkProfileRegistry,
    BenchmarkQuery,
    HitRateCalculator,
    MetricRegistry,
    MRRCalculator,
    NDCGCalculator,
    PrecisionCalculator,
    RecallCalculator,
    RetrievalEvaluator,
    compute_hit_rate,
    compute_mrr,
    compute_ndcg,
    compute_precision,
    compute_recall,
)
from src.core.retrieval.bm25 import (
    BM25CandidateGenerator,
    MetadataDocumentStore,
    WhitespaceTokenizer,
)
from src.core.retrieval.hybrid import HybridRetriever
from src.core.retrieval.retrieval_models import (
    BM25RetrievalDefinition,
    EvidencePassage,
    HybridRetrievalDefinition,
)


@pytest.fixture
def temp_dir() -> typing.Generator[str, None, None]:
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


@pytest.fixture
def sample_passages() -> tuple[EvidencePassage, ...]:
    return (
        EvidencePassage(document_id="doc1", span_id="span-1", text="text 1", score=0.9),
        EvidencePassage(document_id="doc2", span_id="span-2", text="text 2", score=0.8),
        EvidencePassage(document_id="doc3", span_id="span-3", text="text 3", score=0.7),
    )


def test_metric_calculators_mathematical_correctness(
    sample_passages: tuple[EvidencePassage, ...],
) -> None:
    expected_spans = ("span-2", "span-4")

    # top_k = 3
    # Matches: span-2 (at rank 2)
    # Recall = 1 matched / 2 expected = 0.5
    assert compute_recall(sample_passages, expected_spans, top_k=3) == 0.5

    # Precision = 1 matched / 3 retrieved = 0.3333333333333333
    assert (
        pytest.approx(
            compute_precision(sample_passages, expected_spans, top_k=3), 0.001
        )
        == 0.3333
    )

    # MRR = 1/2 = 0.5
    assert compute_mrr(sample_passages, expected_spans, top_k=3) == 0.5

    # Hit rate = 1.0
    assert compute_hit_rate(sample_passages, expected_spans, top_k=3) == 1.0

    # nDCG check
    ndcg = compute_ndcg(sample_passages, expected_spans, top_k=3)
    assert ndcg > 0.0 and ndcg <= 1.0


def test_metric_registry_operations() -> None:
    reg = MetricRegistry(
        calculators={
            "recall": RecallCalculator(),
            "precision": PrecisionCalculator(),
        }
    )

    assert isinstance(reg.resolve("recall"), RecallCalculator)
    assert isinstance(reg.resolve("PRECISION"), PrecisionCalculator)

    with pytest.raises(MetricNotFoundError):
        reg.resolve("non_existent_metric")

    with pytest.raises(DuplicateMetricError):
        MetricRegistry(
            calculators={
                "recall": RecallCalculator(),
                "RECALL": RecallCalculator(),
            }
        )


def test_dataset_fingerprint_reproducibility() -> None:
    q1 = BenchmarkQuery(
        query_id="q1",
        query_text="What is water boiling point?",
        expected_span_ids=("span-1",),
    )
    q2 = BenchmarkQuery(
        query_id="q2",
        query_text="Speed of light in vacuum?",
        expected_span_ids=("span-2",),
    )

    ds1 = BenchmarkDataset(benchmark_id="bm1", dataset_version="1.0", queries=(q1, q2))
    ds2 = BenchmarkDataset(benchmark_id="bm1", dataset_version="1.0", queries=(q1, q2))

    assert ds1.dataset_fingerprint != ""
    assert ds1.dataset_fingerprint == ds2.dataset_fingerprint

    # Modified query text -> different fingerprint
    q1_mod = BenchmarkQuery(
        query_id="q1",
        query_text="What is water boiling point at sea level?",
        expected_span_ids=("span-1",),
    )
    ds3 = BenchmarkDataset(
        benchmark_id="bm1", dataset_version="1.0", queries=(q1_mod, q2)
    )
    assert ds1.dataset_fingerprint != ds3.dataset_fingerprint


def test_benchmark_profile_registry() -> None:
    metric_reg = MetricRegistry(
        calculators={"recall": RecallCalculator(), "precision": PrecisionCalculator()}
    )
    evaluator = RetrievalEvaluator(metric_registry=metric_reg)
    definition = BenchmarkDefinition(top_k=5, metrics=("recall", "precision"))

    profile = BenchmarkProfile(
        profile_id="bm_profile",
        definition=definition,
        strategy=evaluator,
    )
    registry = BenchmarkProfileRegistry(profiles=(profile,))

    assert registry.resolve("bm_profile") is profile

    with pytest.raises(DuplicateBenchmarkProfileError):
        BenchmarkProfileRegistry(profiles=(profile, profile))

    with pytest.raises(BenchmarkProfileNotFoundError):
        registry.resolve("non_existent")


def test_bootstrap_benchmark_registry() -> None:
    config = Settings()
    registry = build_benchmark_registry(config)
    resolved = registry.resolve("default_benchmark")
    assert resolved.definition.top_k == 5


def test_end_to_end_retrieval_benchmarking(temp_dir: str) -> None:
    # 1. Setup DocumentStore & Corpus
    path = os.path.join(temp_dir, "metadata.jsonl")
    chunk1 = Chunk(
        span_id="span-1",
        document_id="doc1",
        text="Water boils at 100 degrees Celsius.",
        start_char=0,
        end_char=35,
        dataset_version="1.0",
        metadata={"corpus_index": 0},
    )
    chunk2 = Chunk(
        span_id="span-2",
        document_id="doc2",
        text="Light travels at 299,792,458 meters per second.",
        start_char=0,
        end_char=47,
        dataset_version="1.0",
        metadata={"corpus_index": 1},
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(chunk1.model_dump_json() + "\n")
        f.write(chunk2.model_dump_json() + "\n")

    doc_store = MetadataDocumentStore(path)
    bm25_index = BM25Okapi(
        [
            WhitespaceTokenizer().tokenize(chunk1.text),
            WhitespaceTokenizer().tokenize(chunk2.text),
        ]
    )
    bm25_gen = BM25CandidateGenerator(
        index=bm25_index,
        span_ids=["span-1", "span-2"],
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

    # 2. Build BenchmarkDataset
    q1 = BenchmarkQuery(
        query_id="q1",
        query_text="Water boiling temperature",
        expected_span_ids=("span-1",),
    )
    q2 = BenchmarkQuery(
        query_id="q2",
        query_text="Speed of light",
        expected_span_ids=("span-2",),
    )
    dataset = BenchmarkDataset(
        benchmark_id="physics_bench", dataset_version="1.0", queries=(q1, q2)
    )

    # 3. Build MetricRegistry and RetrievalEvaluator
    metric_reg = MetricRegistry(
        calculators={
            "recall": RecallCalculator(),
            "precision": PrecisionCalculator(),
            "mrr": MRRCalculator(),
            "ndcg": NDCGCalculator(),
            "hit_rate": HitRateCalculator(),
        }
    )
    evaluator = RetrievalEvaluator(metric_registry=metric_reg)
    bm_def = BenchmarkDefinition(top_k=2)
    ret_def = HybridRetrievalDefinition(
        bm25_definition=BM25RetrievalDefinition(top_k=2), top_k=2
    )
    rr_def = RerankingDefinition(top_k_input=2, top_k_output=2)

    env_metadata = BenchmarkEnvironmentMetadata(
        retrieval_profile_id="hybrid",
        reranking_profile_id="cross_encoder",
        cache_profile_id="in_memory",
        corpus_version="v1.0",
        benchmark_dataset_fingerprint=dataset.dataset_fingerprint,
        execution_device="cpu",
    )

    # 4. Execute Benchmark
    report = evaluator.evaluate_dataset(
        dataset=dataset,
        retriever=cached_retriever,
        definition=bm_def,
        retrieval_definition=ret_def,
        reranking_definition=rr_def,
        environment_metadata=env_metadata,
    )

    assert report.benchmark_id == "physics_bench"
    assert report.total_queries == 2
    assert report.aggregate_metrics.mean_recall == 1.0
    assert report.aggregate_metrics.hit_rate == 1.0
    assert report.latency_summary.mean_ms >= 0.0
    assert report.dataset_fingerprint == dataset.dataset_fingerprint

    # 5. Determinism Assertion: Repeated run produces identical report
    report2 = evaluator.evaluate_dataset(
        dataset=dataset,
        retriever=cached_retriever,
        definition=bm_def,
        retrieval_definition=ret_def,
        reranking_definition=rr_def,
        environment_metadata=env_metadata,
    )
    assert report.aggregate_metrics == report2.aggregate_metrics
