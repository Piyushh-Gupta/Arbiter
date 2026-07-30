"""Concurrency limiter abstraction and implementation for production retrieval optimization."""

import threading
from typing import Any, Protocol, runtime_checkable

from src.core.exceptions import OptimizationTimeoutError


@runtime_checkable
class ConcurrencyLimiter(Protocol):
    """Protocol abstracting synchronization and concurrency control primitives."""

    def acquire(self, timeout_ms: float | None = None) -> bool:
        """Acquires a concurrency slot within the specified timeout in milliseconds."""
        ...

    def release(self) -> None:
        """Releases an acquired concurrency slot."""
        ...

    def __enter__(self) -> "ConcurrencyLimiter":
        """Context manager entry acquiring a slot."""
        ...

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit releasing a slot."""
        ...


class BoundedSemaphoreConcurrencyLimiter(ConcurrencyLimiter):
    """
    Concrete implementation wrapping threading.BoundedSemaphore.
    Ensures bounded concurrency and request isolation without exposing raw primitives.
    """

    def __init__(self, max_concurrency: int) -> None:
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive.")
        self._max_concurrency = max_concurrency
        self._semaphore = threading.BoundedSemaphore(max_concurrency)

    @property
    def max_concurrency(self) -> int:
        return self._max_concurrency

    def acquire(self, timeout_ms: float | None = None) -> bool:
        if timeout_ms is not None:
            timeout_sec = max(0.0, timeout_ms / 1000.0)
            acquired = self._semaphore.acquire(timeout=timeout_sec)
        else:
            acquired = self._semaphore.acquire()
        return acquired

    def release(self) -> None:
        try:
            self._semaphore.release()
        except ValueError as e:
            raise ValueError("Released more times than acquired.") from e

    def __enter__(self) -> "BoundedSemaphoreConcurrencyLimiter":
        if not self.acquire():
            raise OptimizationTimeoutError("Failed to acquire concurrency lock.")
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()
