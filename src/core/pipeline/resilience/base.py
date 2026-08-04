"""Stateless protocols for the resilience subsystem."""

from typing import Any, Callable, Protocol, runtime_checkable

from src.core.pipeline.base import BasePipelineOrchestrator
from src.core.pipeline.pipeline_models import (
    PipelineExecutionRequest,
    PipelineExecutionResult,
)
from src.core.pipeline.resilience.resilience_models import (
    PipelineRecoveryResult,
    RecoveryDefinition,
    RetryDefinition,
    RetryExecutionTrace,
    TimeoutDefinition,
)


@runtime_checkable
class BaseRetryStrategy(Protocol):
    """Stateless protocol defining retry sequence execution."""

    def execute_with_retry(
        self,
        fn: Callable[[], PipelineExecutionResult],
        definition: RetryDefinition,
        execution_id: str,
    ) -> tuple[PipelineExecutionResult, RetryExecutionTrace]:
        """Executes fn with the retry policy from definition.

        Returns:
            A tuple of (successful_result, retry_trace).

        Raises:
            PipelineStageExecutionError: If all attempts fail.
        """
        ...


@runtime_checkable
class BaseTimeoutPolicy(Protocol):
    """Stateless protocol for enforcing per-execution wall-clock time limits."""

    def execute_with_timeout(
        self,
        fn: Callable[[], PipelineExecutionResult],
        definition: TimeoutDefinition,
    ) -> PipelineExecutionResult:
        """Executes fn within the wall-clock limit defined by definition.timeout_ms.

        Raises:
            PipelineResilienceTimeoutError: If the limit is exceeded.
        """
        ...


@runtime_checkable
class BaseRecoveryStrategy(Protocol):
    """Stateless protocol for handling terminal pipeline failures."""

    def validate_compatibility(self, definition: RecoveryDefinition) -> None:
        """Validates that this strategy is compatible with the given definition.

        Raises:
            PipelineResilienceConfigurationError: On incompatibility.
        """
        ...

    def recover(
        self,
        request: PipelineExecutionRequest,
        trace: RetryExecutionTrace,
        definition: RecoveryDefinition,
        pipeline_profile_id: str,
        resilience_profile_id: str,
        timeout_enforced: bool,
    ) -> PipelineRecoveryResult:
        """Executes recovery logic after all retry attempts have been exhausted."""
        ...


@runtime_checkable
class BasePipelineResilienceController(Protocol):
    """Stateless protocol for the resilience coordination layer."""

    def execute(
        self,
        request: PipelineExecutionRequest,
        orchestrator: BasePipelineOrchestrator,
        profile: Any,  # Avoid circular dependency with PipelineResilienceProfile
    ) -> PipelineExecutionResult:
        """Wraps orchestrator.execute(request) with retry, timeout, and recovery."""
        ...
