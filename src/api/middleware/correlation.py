"""Correlation middleware component."""

import uuid

from src.api.middleware.middleware_models import (
    CorrelationContext,
    MiddlewareExecutionContext,
    RequestLifecyclePhase,
)
from src.core.exceptions import InvalidLifecycleTransitionError


class CorrelationComponent:
    """Stateless middleware component for propagating correlation context."""

    def execute(
        self, context: MiddlewareExecutionContext
    ) -> MiddlewareExecutionContext:
        """Extracts or generates correlation ID and updates the context."""
        if context.phase != RequestLifecyclePhase.REQUEST_RECEIVED:
            raise InvalidLifecycleTransitionError(
                f"CorrelationComponent expected phase {RequestLifecyclePhase.REQUEST_RECEIVED.name}, got {context.phase.name}"
            )

        # In a real transport, we would extract from request headers.
        # Since we are keeping middleware transport-only and avoiding side-effects,
        # we generate a correlation ID if one is not present.

        # Example extracting from a standard dictionary request for testing purposes
        correlation_id = None
        client_id = None
        if isinstance(context.request, dict):
            headers = context.request.get("headers", {})
            correlation_id = headers.get("X-Correlation-ID")
            client_id = headers.get("X-Client-ID")

        if not correlation_id:
            correlation_id = str(uuid.uuid4())

        correlation_context = CorrelationContext(
            correlation_id=correlation_id,
            client_id=client_id,
        )

        return context.model_copy(
            update={
                "correlation_context": correlation_context,
                "phase": RequestLifecyclePhase.CORRELATION_ESTABLISHED,
            }
        )
