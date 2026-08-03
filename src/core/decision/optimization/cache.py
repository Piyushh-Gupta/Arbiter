"""Thread-safe cache implementations for Decision Engine (M4.7)."""

import time
from collections import OrderedDict
from threading import Lock
from typing import Protocol, runtime_checkable

from src.core.decision.decision_models import DecisionResult


@runtime_checkable
class BaseDecisionCache(Protocol):
    """Protocol defining post-decision execution cache backends."""

    def get(self, key: str) -> DecisionResult | None:
        """Retrieves result from the cache. Returns None on miss or expiration."""
        ...

    def put(self, key: str, value: DecisionResult) -> None:
        """Saves result to the cache."""
        ...

    def contains(self, key: str) -> bool:
        """Checks if a key is present and not expired."""
        ...

    def invalidate(self, key: str) -> None:
        """Removes a specific key from the cache."""
        ...

    def clear(self) -> None:
        """Purges all entries from the cache."""
        ...


class InMemoryDecisionCache(BaseDecisionCache):
    """Thread-safe, LRU evicted, in-memory cache implementation with TTL support."""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300) -> None:
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, tuple[DecisionResult, float]] = OrderedDict()
        self._lock = Lock()

    def get(self, key: str) -> DecisionResult | None:
        with self._lock:
            if key not in self._cache:
                return None
            val, entry_time = self._cache[key]
            # TTL Expiry check
            if time.time() - entry_time > self._ttl_seconds:
                del self._cache[key]
                return None
            # Move to end (MRU)
            self._cache.move_to_end(key)
            return val

    def put(self, key: str, value: DecisionResult) -> None:
        with self._lock:
            # If key already exists, update and move to end
            if key in self._cache:
                self._cache[key] = (value, time.time())
                self._cache.move_to_end(key)
                return

            # Check max size limit and evict LRU
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)

            self._cache[key] = (value, time.time())

    def contains(self, key: str) -> bool:
        with self._lock:
            if key not in self._cache:
                return False
            _, entry_time = self._cache[key]
            if time.time() - entry_time > self._ttl_seconds:
                del self._cache[key]
                return False
            return True

    def invalidate(self, key: str) -> None:
        with self._lock:
            if key in self._cache:
                del self._cache[key]

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
