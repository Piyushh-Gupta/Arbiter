import hashlib
from datetime import datetime, timezone
from typing import Any

from src.core.exceptions import PipelineStageExecutionError
from src.core.pipeline.base import BasePipelineOrchestrator
from src.core.pipeline.pipeline_models import (
    PipelineExecutionRequest,
    PipelineExecutionResult,
)
from src.core.pipeline.resilience.base import (
    BasePipelineResilienceController,
    BaseRetryStrategy,
    BaseTimeoutPolicy,
)
from src.core.pipeline.resilience.resilience_models import (
    ResilienceRuntimeMetadata,
    RetryExecutionTrace,
)


class PipelineResilienceController(BasePipelineResilienceController):
    """Coordinates retry, timeout, and recovery execution for a pipeline execution."""

    def __init__(
        self,
        retry_strategy: BaseRetryStrategy,
        timeout_policy: BaseTimeoutPolicy,
    ) -> None:
        self._retry_strategy = retry_strategy
        self._timeout_policy = timeout_policy

    def execute(
        self,
        request: PipelineExecutionRequest,
        orchestrator: BasePipelineOrchestrator,
        profile: Any,  # PipelineResilienceProfile
    ) -> PipelineExecutionResult:
        """Executes the orchestrator with configured retry, timeout, and recovery policies."""
        definition = profile.definition
        if not definition.enabled:
            return orchestrator.execute(request)

        execution_id = hashlib.sha256(
            f"{request.pipeline_profile_id}{request.claim}".encode("utf-8")
        ).hexdigest()

        def execute_pipeline() -> PipelineExecutionResult:
            return orchestrator.execute(request)

        def execute_with_timeout() -> PipelineExecutionResult:
            return self._timeout_policy.execute_with_timeout(
                execute_pipeline, definition.timeout
            )

        retry_trace = None
        succeeded = False
        result = None
        terminal_error_exc = None

        try:
            result, retry_trace = self._retry_strategy.execute_with_retry(
                execute_with_timeout, definition.retry, execution_id
            )
            succeeded = True
        except PipelineStageExecutionError as e:
            retry_trace = getattr(e, "retry_trace", None)
            if retry_trace is None:
                retry_trace = RetryExecutionTrace(
                    execution_id=execution_id,
                    total_attempts=0,
                    succeeded=False,
                    attempts=(),
                    total_retry_overhead_ms=0.0,
                    terminal_error=str(e),
                )
            terminal_error_exc = e

        if succeeded and result is not None:
            assert retry_trace is not None
            meta = ResilienceRuntimeMetadata(
                pipeline_profile_id=request.pipeline_profile_id,
                resilience_profile_id=profile.profile_id,
                timeout_enforced=definition.timeout.enabled,
                retry_trace=retry_trace,
                recovery_invoked=False,
                recovery_strategy_id=None,
                observed_at=datetime.now(timezone.utc),
            )
            object.__setattr__(result, "resilience_metadata", meta)
            return result

        assert retry_trace is not None
        recovery_def = definition.recovery

        if recovery_def.enabled:
            try:
                recovery_result = profile.recovery_strategy.recover(
                    request,
                    retry_trace,
                    recovery_def,
                    pipeline_profile_id=request.pipeline_profile_id,
                    resilience_profile_id=profile.profile_id,
                    timeout_enforced=definition.timeout.enabled,
                )
                if recovery_result.succeeded and recovery_result.result is not None:
                    res_val = recovery_result.result
                    assert isinstance(res_val, PipelineExecutionResult)
                    object.__setattr__(
                        res_val,
                        "resilience_metadata",
                        recovery_result.resilience_metadata,
                    )
                    return res_val
                else:
                    terminal_error_msg = (
                        recovery_result.failure_reason or "Recovery strategy failed"
                    )
                    exc = PipelineStageExecutionError(
                        f"Pipeline execution failed. Recovery strategy '{recovery_def.strategy_id}' failed: {terminal_error_msg}"
                    )
                    object.__setattr__(
                        exc, "resilience_metadata", recovery_result.resilience_metadata
                    )
                    raise exc from terminal_error_exc
            except Exception as rec_err:
                if isinstance(rec_err, PipelineStageExecutionError) and hasattr(
                    rec_err, "resilience_metadata"
                ):
                    raise rec_err
                meta = ResilienceRuntimeMetadata(
                    pipeline_profile_id=request.pipeline_profile_id,
                    resilience_profile_id=profile.profile_id,
                    timeout_enforced=definition.timeout.enabled,
                    retry_trace=retry_trace,
                    recovery_invoked=True,
                    recovery_strategy_id=recovery_def.strategy_id,
                    observed_at=datetime.now(timezone.utc),
                )
                exc = PipelineStageExecutionError(
                    f"Pipeline execution failed. Recovery strategy '{recovery_def.strategy_id}' encountered error: {rec_err}"
                )
                object.__setattr__(exc, "resilience_metadata", meta)
                raise exc from terminal_error_exc
        else:
            meta = ResilienceRuntimeMetadata(
                pipeline_profile_id=request.pipeline_profile_id,
                resilience_profile_id=profile.profile_id,
                timeout_enforced=definition.timeout.enabled,
                retry_trace=retry_trace,
                recovery_invoked=False,
                recovery_strategy_id=None,
                observed_at=datetime.now(timezone.utc),
            )
            assert terminal_error_exc is not None
            object.__setattr__(terminal_error_exc, "resilience_metadata", meta)
            raise terminal_error_exc
