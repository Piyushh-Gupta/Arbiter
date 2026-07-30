"""Stateless protocols for the Retrieval Caching subsystem."""

from typing import Protocol, runtime_checkable

from src.core.cache.cache_models import CacheEntry


@runtime_checkable
class BaseRetrievalCache(Protocol):
    """Stateless protocol for retrieval result caching backends."""

    def get(self, key: str) -> CacheEntry | None:
        """
        Retrieves a cached CacheEntry by key.
        Returns None if key is missing or entry has expired.
        """
        ...

    def put(self, entry: CacheEntry) -> None:
        """
        Stores an immutable CacheEntry under its cache_key.
        """
        ...

    def contains(self, key: str) -> bool:
        """
        Checks if a valid, unexpired entry exists for the given key.
        """
        ...

    def invalidate(self, key: str) -> bool:
        """
        Evicts a specific entry from the cache. Returns True if evicted, False otherwise.
        """
        ...

    def clear(self) -> None:
        """
        Flushes all entries from the cache.
        """
        ...
