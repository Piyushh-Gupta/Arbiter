"""Concrete implementations of retrieval caching strategies."""

import time
from collections import OrderedDict
from threading import Lock
from typing import Any

from src.core.cache.base import BaseRetrievalCache
from src.core.cache.cache_models import CacheDefinition, CacheEntry, CacheKeyGenerator
from src.core.exceptions import CacheConfigurationError, CacheExecutionError
from src.core.reranking.base import BaseReranker
from src.core.retrieval.base import BaseRetriever
from src.core.retrieval.retrieval_models import EvidenceBundle


class InMemoryRetrievalCache(BaseRetrievalCache):
    """
    Thread-safe, bounded in-memory cache strategy supporting LRU/FIFO eviction and TTL expiration.
    """

    def __init__(self, definition: CacheDefinition) -> None:
        if not isinstance(definition, CacheDefinition):
            raise CacheConfigurationError(
                f"InMemoryRetrievalCache requires CacheDefinition, got {type(definition).__name__}"
            )
        self._definition = definition
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = Lock()

    @property
    def definition(self) -> CacheDefinition:
        return self._definition

    def get(self, key: str) -> CacheEntry | None:
        if not self._definition.enabled:
            return None

        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None

            now = time.time()
            if entry.expires_at is not None and now >= entry.expires_at:
                self._store.pop(key, None)
                return None

            if self._definition.eviction_policy == "lru":
                self._store.move_to_end(key)

            return entry

    def put(self, entry: CacheEntry) -> None:
        if not self._definition.enabled:
            return

        if not isinstance(entry, CacheEntry):
            raise CacheExecutionError("InMemoryRetrievalCache put requires CacheEntry.")

        with self._lock:
            now = time.time()
            if entry.expires_at is not None and now >= entry.expires_at:
                return

            if entry.cache_key in self._store:
                self._store.pop(entry.cache_key)
            elif len(self._store) >= self._definition.max_entries:
                self._store.popitem(last=False)  # Evict oldest / LRU item

            self._store[entry.cache_key] = entry

    def contains(self, key: str) -> bool:
        return self.get(key) is not None

    def invalidate(self, key: str) -> bool:
        with self._lock:
            if key in self._store:
                self._store.pop(key)
                return True
            return False

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


class CachedRetriever:
    """
    Stateless proxy wrapper orchestrating retrieval caching, key generation, and underlying execution.
    """

    def __init__(
        self,
        retriever: BaseRetriever | Any,
        cache: BaseRetrievalCache,
        key_generator: CacheKeyGenerator,
        definition: CacheDefinition,
        reranker: BaseReranker | Any | None = None,
        corpus_version: str = "default",
    ) -> None:
        self._retriever = retriever
        self._cache = cache
        self._key_generator = key_generator
        self._definition = definition
        self._reranker = reranker
        self._corpus_version = corpus_version

    def retrieve(
        self,
        claim: str,
        retrieval_profile_id: str = "default",
        reranking_profile_id: str | None = None,
        retrieval_definition: Any | None = None,
        reranking_definition: Any | None = None,
    ) -> EvidenceBundle:
        """
        Executes retrieval with deterministic caching.
        """
        key: str | None = None
        if self._definition.enabled:
            key = self._key_generator.generate_key(
                query=claim,
                retrieval_profile_id=retrieval_profile_id,
                reranking_profile_id=reranking_profile_id,
                retrieval_definition=retrieval_definition,
                reranking_definition=reranking_definition,
                corpus_version=self._corpus_version,
                cache_schema_version=self._definition.cache_schema_version,
            )
            cached_entry = self._cache.get(key)
            if cached_entry is not None:
                return cached_entry.evidence_bundle

        # Cache miss or disabled — execute underlying retriever
        if hasattr(self._retriever, "retrieve"):
            retriever_func = getattr(self._retriever, "retrieve")
            if retrieval_definition is not None:
                bundle = retriever_func(claim, retrieval_definition)
            else:
                bundle = retriever_func(claim)
        elif callable(self._retriever):
            bundle = self._retriever(claim)
        else:
            raise CacheExecutionError(
                "Underlying retriever does not implement retrieve."
            )

        # Apply reranker if provided and bundle was produced
        if self._reranker is not None and reranking_definition is not None:
            if hasattr(self._reranker, "rerank"):
                bundle = self._reranker.rerank(claim, bundle, reranking_definition)

        # Store in cache
        if self._definition.enabled and key is not None:
            now = time.time()
            expires = now + self._definition.ttl_seconds
            entry = CacheEntry(
                cache_key=key,
                created_at=now,
                expires_at=expires,
                corpus_version=self._corpus_version,
                cache_schema_version=self._definition.cache_schema_version,
                evidence_bundle=bundle,
            )
            self._cache.put(entry)

        if not isinstance(bundle, EvidenceBundle):
            raise CacheExecutionError(
                f"Retriever returned invalid output type {type(bundle).__name__}, expected EvidenceBundle."
            )

        return bundle
