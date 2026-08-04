"""Fixed retry strategy implementation."""

import logging
import time
from typing import Callable

from src.core.exceptions import PipelineStageExecutionError
from src.core.pipeline.pipeline_models import PipelineExecutionResult
from src.core.pipeline.resilience.base import BaseRetryStrategy
from src.core.pipeline.resilience.resilience_models import (
    RetryAttemptRecord,
    RetryDefinition,
    RetryExecutionTrace,
)

logger = logging.getLogger("arbiter.resilience")


class FixedRetryStrategy(BaseRetryStrategy):
    """Stateless retry strategy using fixed inter-attempt delay."""

    def __init__(
        self,
        retryable_types: tuple[type[BaseException], ...] = (
            PipelineStageExecutionError,
        ),
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._retryable_types = retryable_types
        self._sleeper = sleeper

    def execute_with_retry(
        self,
        fn: Callable[[], PipelineExecutionResult],
        definition: RetryDefinition,
        execution_id: str,
    ) -> tuple[PipelineExecutionResult, RetryExecutionTrace]:
        """Executes fn with fixed retries and returns the successful result and trace."""
        attempts = []
        start_time = time.perf_counter()
        terminal_error = None
        succeeded = False
        result = None

        for attempt in range(1, definition.max_attempts + 1):
            attempt_start = time.perf_counter()
            try:
                result = fn()
                succeeded = True
                break
            except Exception as e:
                # Type-safe check using resolved types
                if not isinstance(e, self._retryable_types):
                    # Propagate non-retryable exception immediately
                    raise e

                attempt_latency = (time.perf_counter() - attempt_start) * 1000
                attempts.append(
                    RetryAttemptRecord(
                        attempt_number=attempt,
                        error_type=e.__class__.__name__,
                        error_message=str(e),
                        latency_ms=attempt_latency,
                    )
                )
                terminal_error = f"{e.__class__.__name__}: {str(e)}"

                # Apply delay if we have remaining attempts
                if attempt < definition.max_attempts:
                    self._sleeper(definition.retry_delay_ms / 1000.0)

        total_overhead = (time.perf_counter() - start_time) * 1000

        if not succeeded:
            trace = RetryExecutionTrace(
                execution_id=execution_id,
                total_attempts=len(attempts),
                succeeded=False,
                attempts=tuple(attempts),
                total_retry_overhead_ms=total_overhead,
                terminal_error=terminal_error,
            )
            exc = PipelineStageExecutionError(
                f"All retry attempts exhausted: {terminal_error}"
            )
            # Attach the trace to the exception so the controller can retrieve it
            exc.retry_trace = trace  # type: ignore
            raise exc

        assert result is not None

        trace = RetryExecutionTrace(
            execution_id=execution_id,
            total_attempts=len(attempts) + 1,
            succeeded=True,
            attempts=tuple(attempts),
            total_retry_overhead_ms=total_overhead,
            terminal_error=None,
        )

        return result, trace
