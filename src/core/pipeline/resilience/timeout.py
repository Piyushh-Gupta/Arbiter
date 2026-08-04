"""Timeout policy implementation."""

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Callable

from src.core.exceptions import PipelineResilienceTimeoutError
from src.core.pipeline.pipeline_models import PipelineExecutionResult
from src.core.pipeline.resilience.base import BaseTimeoutPolicy
from src.core.pipeline.resilience.resilience_models import TimeoutDefinition

logger = logging.getLogger("arbiter.resilience")


class ThreadPoolTimeoutPolicy(BaseTimeoutPolicy):
    """Timeout policy enforcing time limits via a shared ThreadPoolExecutor."""

    def __init__(self, executor: ThreadPoolExecutor) -> None:
        self._executor = executor

    def execute_with_timeout(
        self,
        fn: Callable[[], PipelineExecutionResult],
        definition: TimeoutDefinition,
    ) -> PipelineExecutionResult:
        """Executes fn within the configured time limit or raises timeout error."""
        if not definition.enabled:
            return fn()

        future = self._executor.submit(fn)
        try:
            return future.result(timeout=definition.timeout_ms / 1000.0)
        except TimeoutError as e:
            raise PipelineResilienceTimeoutError(
                f"Pipeline execution timed out after {definition.timeout_ms}ms"
            ) from e
