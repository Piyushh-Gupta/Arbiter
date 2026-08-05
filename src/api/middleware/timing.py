"""Timing middleware component."""

from src.api.middleware.base import Clock
from src.api.middleware.middleware_models import (
    MiddlewareExecutionContext,
    RequestLifecyclePhase,
)
from src.core.exceptions import InvalidLifecycleTransitionError


class TimingComponent:
    """Stateless middleware component for recording request timing."""

    def __init__(self, clock: Clock) -> None:
        """Initializes the TimingComponent with a deterministic clock."""
        self._clock = clock

    def execute(
        self, context: MiddlewareExecutionContext
    ) -> MiddlewareExecutionContext:
        """Records request start time if in REQUEST_RECEIVED phase."""
        if context.phase not in (
            RequestLifecyclePhase.REQUEST_RECEIVED,
            RequestLifecyclePhase.CORRELATION_ESTABLISHED,
        ):
            raise InvalidLifecycleTransitionError(
                f"TimingComponent cannot start timing from phase {context.phase.name}"
            )

        updated_timing = context.timing.model_copy(
            update={"start_time_ns": self._clock.now_ns()}
        )
        return context.model_copy(update={"timing": updated_timing})
