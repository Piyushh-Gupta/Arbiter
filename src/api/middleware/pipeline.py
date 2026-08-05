"""Middleware pipeline for ordered execution."""

from typing import Sequence

from src.api.middleware.base import BaseMiddlewareComponent
from src.api.middleware.middleware_models import MiddlewareExecutionContext
from src.core.exceptions import MiddlewareConfigurationError


class MiddlewarePipeline:
    """Deterministic pipeline for executing middleware sequentially."""

    def __init__(self, components: Sequence[BaseMiddlewareComponent]) -> None:
        """Initializes the pipeline with a strictly ordered sequence of components."""
        if not components:
            raise MiddlewareConfigurationError(
                "MiddlewarePipeline requires at least one component."
            )
        self._components = tuple(components)

    def execute(
        self, context: MiddlewareExecutionContext
    ) -> MiddlewareExecutionContext:
        """Executes the middleware sequence deterministically.

        Args:
            context: The initial middleware execution context.

        Returns:
            The finalized middleware execution context after all components have run.
        """
        current_context = context
        for component in self._components:
            current_context = component.execute(current_context)
        return current_context
