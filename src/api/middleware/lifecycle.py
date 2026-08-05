"""Lifecycle manager for the API middleware."""

from typing import Any

from src.api.middleware.base import Clock
from src.api.middleware.middleware_models import (
    MiddlewareExecutionContext,
    RequestLifecyclePhase,
)
from src.api.middleware.pipeline import MiddlewarePipeline
from src.core.exceptions import InvalidLifecycleTransitionError


class LifecycleManager:
    """Orchestrator for the request lifecycle."""

    def __init__(
        self,
        pipeline: MiddlewarePipeline,
        clock: Clock,
        active_profile_id: str,
    ) -> None:
        """Initializes the LifecycleManager with a pipeline and clock."""
        self._pipeline = pipeline
        self._clock = clock
        self._active_profile_id = active_profile_id

    def initialize_request(self, request: Any) -> MiddlewareExecutionContext:
        """Initializes the base context for a new incoming request and runs the pipeline."""
        context = MiddlewareExecutionContext(
            request=request,
            contract_profile_id=self._active_profile_id,
            phase=RequestLifecyclePhase.REQUEST_RECEIVED,
        )
        return self._pipeline.execute(context)

    def finalize_request(
        self, context: MiddlewareExecutionContext
    ) -> MiddlewareExecutionContext:
        """Finalizes the request context before returning a response."""
        if context.phase == RequestLifecyclePhase.FINALIZED:
            raise InvalidLifecycleTransitionError("Request is already finalized.")

        updated_timing = context.timing.model_copy(
            update={"end_time_ns": self._clock.now_ns()}
        )
        return context.model_copy(
            update={
                "timing": updated_timing,
                "phase": RequestLifecyclePhase.FINALIZED,
            }
        )
