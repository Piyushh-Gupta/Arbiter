"""Base protocols for the API middleware and request lifecycle."""

from typing import Any, Protocol, runtime_checkable

from src.api.contracts.error_models import ErrorEnvelope
from src.api.middleware.middleware_models import MiddlewareExecutionContext


@runtime_checkable
class Clock(Protocol):
    """Protocol for a deterministic clock."""

    def now_ns(self) -> int:
        """Returns the current time in nanoseconds."""
        ...


@runtime_checkable
class BaseMiddlewareComponent(Protocol):
    """Protocol for a stateless middleware component."""

    def execute(
        self, context: MiddlewareExecutionContext
    ) -> MiddlewareExecutionContext:
        """Executes the middleware logic against the provided context.

        Must return a new updated MiddlewareExecutionContext instance,
        preserving immutability.
        """
        ...


@runtime_checkable
class BaseExceptionTranslator(Protocol):
    """Protocol for translating exceptions into API standard models."""

    def translate(self, exception: Exception) -> ErrorEnvelope:
        """Translates a domain or unhandled exception into an ErrorEnvelope."""
        ...


@runtime_checkable
class BaseLifecycleManager(Protocol):
    """Protocol for request lifecycle orchestration."""

    def initialize_request(self, request: Any) -> MiddlewareExecutionContext:
        """Initializes the base context for a new incoming request."""
        ...

    def finalize_request(
        self, context: MiddlewareExecutionContext
    ) -> MiddlewareExecutionContext:
        """Finalizes the request context before returning a response."""
        ...
