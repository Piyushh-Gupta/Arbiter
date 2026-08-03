"""Stateless execution guard enforcing retry, timeout, and fallback routing (M4.7)."""

import concurrent.futures
from typing import Any, Callable, TypeVar

from src.core.decision.decision_models import (
    DecisionMetadata,
    DecisionResult,
    DecisionTrace,
)
from src.core.decision.optimization.optimization_models import (
    DecisionExecutionGuardDefinition,
)

T = TypeVar("T")


class DecisionExecutionGuard:
    """Execution guard ensuring deterministic timeouts, retries, and fallback results."""

    def __init__(self, definition: DecisionExecutionGuardDefinition) -> None:
        self._definition = definition

    def execute(
        self, func: Callable[..., DecisionResult], *args: Any, **kwargs: Any
    ) -> DecisionResult:
        """Executes the callable with timeouts and retries, falling back on exhaustion."""
        timeout_ms = self._definition.timeout_ms
        max_retries = self._definition.max_retries
        fallback_action = self._definition.fallback_action

        last_error: Any = None

        for attempt in range(max_retries + 1):
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(func, *args, **kwargs)
                    return future.result(timeout=timeout_ms / 1000.0)
            except concurrent.futures.TimeoutError:
                from src.core.exceptions import DecisionExecutionTimeoutError

                last_error = DecisionExecutionTimeoutError(
                    f"Decision execution timed out after {timeout_ms}ms (attempt {attempt + 1})"
                )
            except Exception as e:
                last_error = e

        # Fallback Result construction on persistent errors
        meta = DecisionMetadata(
            strategy_id="execution_guard_fallback",
            configuration_fingerprint="fallback",
        )
        trace = DecisionTrace(
            selected_rule="fallback_guard",
            policy_path=(f"Fallback routed due to persistent error: {last_error}",),
        )
        return DecisionResult(
            final_verdict=fallback_action,
            final_confidence=0.0,
            final_uncertainty=1.0,
            decision_trace=trace,
            metadata=meta,
        )
