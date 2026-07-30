"""Comprehensive unit tests for C1.8 Retrieval Caching."""

import os
import shutil
import tempfile
import time
import typing

import pytest
from pydantic import ValidationError
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from src.core.bootstrap import (
    DummyCrossEncoderScorer,
    DummyNLIModel,
    build_cache_registry,
)
from src.core.cache import (
    BaseRetrievalCache,
    CacheDefinition,
    CachedRetriever,
    CacheEntry,
    CacheKeyGenerator,
    InMemoryRetrievalCache,
    RetrievalCacheProfile,
    RetrievalCacheProfileRegistry,
)
from src.core.config import Settings
from src.core.exceptions import CacheProfileNotFoundError, DuplicateCacheProfileError
from src.core.indexing.models import Chunk
from src.core.reranking.implementations import CrossEncoderReranker
from src.core.reranking.reranking_models import RerankingDefinition
from src.core.retrieval.bm25 import (
    BM25CandidateGenerator,
    MetadataDocumentStore,
    WhitespaceTokenizer,
)
from src.core.retrieval.hybrid import HybridRetriever
from src.core.retrieval.retrieval_models import (
    BM25RetrievalDefinition,
    EvidenceBundle,
    EvidencePassage,
    HybridRetrievalDefinition,
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
def sample_bundle() -> EvidenceBundle:
    return EvidenceBundle(
        claim="Sample claim text",
        passages=(
            EvidencePassage(
                document_id="doc1",
                span_id="span-1",
                text="Sample passage 1 text",
                score=0.95,
            ),
        ),
        metadata=RetrievalMetadata(strategy_id="test", top_k=1),
    )


def test_cache_definition_validation() -> None:
    # Invalid ttl
    with pytest.raises(ValidationError):
        CacheDefinition(ttl_seconds=0)

    # Invalid max_entries
    with pytest.raises(ValidationError):
        CacheDefinition(max_entries=-1)

    # Invalid backend
    with pytest.raises(ValidationError):
        CacheDefinition(backend="invalid_backend")

    # Invalid eviction policy
    with pytest.raises(ValidationError):
        CacheDefinition(eviction_policy="invalid_policy")

    # Valid definition
    c_def = CacheDefinition(
        enabled=True,
        backend="in_memory",
        ttl_seconds=1800,
        max_entries=500,
        eviction_policy="lru",
        cache_schema_version="1.0",
    )
    assert c_def.ttl_seconds == 1800
    assert c_def.max_entries == 500


def test_cache_key_generator_determinism() -> None:
    gen = CacheKeyGenerator()

    key1 = gen.generate_key(
        query="  Climate Change impact  ",
        retrieval_profile_id="hybrid",
        reranking_profile_id="cross_encoder",
        retrieval_definition={"top_k": 5},
        reranking_definition={"top_k_output": 3},
        corpus_version="v1.0",
        cache_schema_version="1.0",
    )

    key2 = gen.generate_key(
        query="climate change impact",
        retrieval_profile_id="hybrid",
        reranking_profile_id="cross_encoder",
        retrieval_definition={"top_k": 5},
        reranking_definition={"top_k_output": 3},
        corpus_version="v1.0",
        cache_schema_version="1.0",
    )

    assert key1 == key2

    # Changed query -> different key
    key3 = gen.generate_key(
        query="Global warming impact",
        retrieval_profile_id="hybrid",
        corpus_version="v1.0",
    )
    assert key1 != key3

    # Changed corpus version -> different key
    key4 = gen.generate_key(
        query="climate change impact",
        retrieval_profile_id="hybrid",
        reranking_profile_id="cross_encoder",
        retrieval_definition={"top_k": 5},
        reranking_definition={"top_k_output": 3},
        corpus_version="v2.0",
        cache_schema_version="1.0",
    )
    assert key1 != key4

    # Changed schema version -> different key
    key5 = gen.generate_key(
        query="climate change impact",
        retrieval_profile_id="hybrid",
        reranking_profile_id="cross_encoder",
        retrieval_definition={"top_k": 5},
        reranking_definition={"top_k_output": 3},
        corpus_version="v1.0",
        cache_schema_version="2.0",
    )
    assert key1 != key5


def test_in_memory_cache_operations(sample_bundle: EvidenceBundle) -> None:
    c_def = CacheDefinition(max_entries=2, ttl_seconds=10)
    cache = InMemoryRetrievalCache(c_def)

    now = time.time()
    entry1 = CacheEntry(
        cache_key="key1",
        created_at=now,
        expires_at=now + 10,
        corpus_version="v1",
        cache_schema_version="1.0",
        evidence_bundle=sample_bundle,
    )

    cache.put(entry1)
    assert cache.contains("key1")
    retrieved = cache.get("key1")
    assert retrieved is not None
    assert retrieved.evidence_bundle == sample_bundle

    # Invalidate key1
    assert cache.invalidate("key1")
    assert not cache.contains("key1")
    assert cache.get("key1") is None


def test_cache_lru_eviction(sample_bundle: EvidenceBundle) -> None:
    c_def = CacheDefinition(max_entries=2, ttl_seconds=100)
    cache = InMemoryRetrievalCache(c_def)
    now = time.time()

    e1 = CacheEntry(
        cache_key="k1",
        created_at=now,
        corpus_version="v1",
        cache_schema_version="1.0",
        evidence_bundle=sample_bundle,
    )
    e2 = CacheEntry(
        cache_key="k2",
        created_at=now,
        corpus_version="v1",
        cache_schema_version="1.0",
        evidence_bundle=sample_bundle,
    )
    e3 = CacheEntry(
        cache_key="k3",
        created_at=now,
        corpus_version="v1",
        cache_schema_version="1.0",
        evidence_bundle=sample_bundle,
    )

    cache.put(e1)
    cache.put(e2)
    # Access k1 to make k2 the LRU item
    _ = cache.get("k1")

    # Put e3 -> evicts k2
    cache.put(e3)

    assert cache.contains("k1")
    assert not cache.contains("k2")
    assert cache.contains("k3")


def test_cache_ttl_expiration(sample_bundle: EvidenceBundle) -> None:
    c_def = CacheDefinition(max_entries=10, ttl_seconds=1)
    cache = InMemoryRetrievalCache(c_def)
    now = time.time()

    # Expired entry (expires_at in past)
    expired_entry = CacheEntry(
        cache_key="exp_key",
        created_at=now - 5,
        expires_at=now - 1,
        corpus_version="v1",
        cache_schema_version="1.0",
        evidence_bundle=sample_bundle,
    )
    cache.put(expired_entry)

    assert cache.get("exp_key") is None
    assert not cache.contains("exp_key")


class CountedRetriever:
    def __init__(self, bundle: EvidenceBundle) -> None:
        self.bundle = bundle
        self.call_count = 0

    def retrieve(self, claim: str) -> EvidenceBundle:
        self.call_count += 1
        return self.bundle


def test_cached_retriever_hits_and_misses(sample_bundle: EvidenceBundle) -> None:
    retriever = CountedRetriever(sample_bundle)
    c_def = CacheDefinition(enabled=True, ttl_seconds=100)
    cache = InMemoryRetrievalCache(c_def)
    key_gen = CacheKeyGenerator()

    cached_service = CachedRetriever(
        retriever=retriever,
        cache=cache,
        key_generator=key_gen,
        definition=c_def,
        corpus_version="v1",
    )

    # 1. First invocation -> Miss -> Call underlying retriever
    res1 = cached_service.retrieve("Claim text", retrieval_profile_id="p1")
    assert res1 == sample_bundle
    assert retriever.call_count == 1

    # 2. Second invocation -> Hit -> Return cached bundle without calling retriever
    res2 = cached_service.retrieve("Claim text", retrieval_profile_id="p1")
    assert res2 == sample_bundle
    assert retriever.call_count == 1


def test_cache_profile_registry() -> None:
    c_def = CacheDefinition()
    cache = InMemoryRetrievalCache(c_def)
    profile = RetrievalCacheProfile(
        profile_id="default_cache",
        definition=c_def,
        strategy=cache,
    )

    registry = RetrievalCacheProfileRegistry(profiles=(profile,))
    assert registry.resolve("default_cache") is profile

    with pytest.raises(DuplicateCacheProfileError):
        RetrievalCacheProfileRegistry(profiles=(profile, profile))

    with pytest.raises(CacheProfileNotFoundError):
        registry.resolve("non_existent")


def test_bootstrap_cache_registry() -> None:
    config = Settings()
    registry = build_cache_registry(config)
    resolved = registry.resolve("default_cache")
    assert resolved.definition.enabled is True
    assert isinstance(resolved.strategy, BaseRetrievalCache)


def test_end_to_end_pipeline_caching_parity(temp_dir: str) -> None:
    # Build indexing data and DocumentStore
    path = os.path.join(temp_dir, "metadata.jsonl")
    chunk = Chunk(
        span_id="span-1",
        document_id="doc1",
        text="Water boils at 100 degrees Celsius at sea level.",
        start_char=0,
        end_char=48,
        dataset_version="1.0",
        metadata={"corpus_index": 0},
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(chunk.model_dump_json() + "\n")

    doc_store = MetadataDocumentStore(path)
    bm25_index = BM25Okapi([WhitespaceTokenizer().tokenize(chunk.text)])
    bm25_gen = BM25CandidateGenerator(
        index=bm25_index,
        span_ids=["span-1"],
        tokenizer=WhitespaceTokenizer(),
    )
    hybrid_retriever = HybridRetriever(
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
        retriever=hybrid_retriever,
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

    # First execution (Cache Miss)
    bundle_uncached = cached_retriever.retrieve(
        claim="Boiling point of water",
        retrieval_profile_id="hybrid",
        reranking_profile_id="cross_encoder",
        retrieval_definition=ret_def,
        reranking_definition=rr_def,
    )

    # Second execution (Cache Hit)
    bundle_cached = cached_retriever.retrieve(
        claim="Boiling point of water",
        retrieval_profile_id="hybrid",
        reranking_profile_id="cross_encoder",
        retrieval_definition=ret_def,
        reranking_definition=rr_def,
    )

    # Parity check: Cached and Uncached outputs are identical
    assert bundle_uncached == bundle_cached

    # Feed bundle into downstream NLIVerifier
    verifier = NLIVerifier(model=DummyNLIModel(), strategy_id="dummy_nli")
    ver_def = NLIVerificationDefinition(top_k=1)

    ver_res = verifier.verify("Boiling point of water", bundle_cached, ver_def)
    assert ver_res.label.name == "SUPPORTS"
