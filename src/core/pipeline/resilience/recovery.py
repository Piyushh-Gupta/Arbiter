"""Recovery strategy implementations."""

import logging
from datetime import datetime, timezone

from src.core.exceptions import PipelineResilienceConfigurationError
from src.core.pipeline.pipeline_models import PipelineExecutionRequest
from src.core.pipeline.resilience.base import BaseRecoveryStrategy
from src.core.pipeline.resilience.resilience_models import (
    PipelineRecoveryResult,
    RecoveryDefinition,
    ResilienceRuntimeMetadata,
    RetryExecutionTrace,
)

logger = logging.getLogger("arbiter.resilience")


class NullRecoveryStrategy(BaseRecoveryStrategy):
    """Fallback recovery strategy that returns failed recovery result."""

    def validate_compatibility(self, definition: RecoveryDefinition) -> None:
        """Validates configuration compatibility."""
        if not isinstance(definition, RecoveryDefinition):
            raise PipelineResilienceConfigurationError("Invalid definition type.")

    def recover(
        self,
        request: PipelineExecutionRequest,
        trace: RetryExecutionTrace,
        definition: RecoveryDefinition,
        pipeline_profile_id: str,
        resilience_profile_id: str,
        timeout_enforced: bool,
    ) -> PipelineRecoveryResult:
        """Returns failed recovery result with terminal error from trace."""
        meta = ResilienceRuntimeMetadata(
            pipeline_profile_id=pipeline_profile_id,
            resilience_profile_id=resilience_profile_id,
            timeout_enforced=timeout_enforced,
            retry_trace=trace,
            recovery_invoked=True,
            recovery_strategy_id=definition.strategy_id,
            observed_at=datetime.now(timezone.utc),
        )
        return PipelineRecoveryResult(
            recovery_strategy_id=definition.strategy_id,
            succeeded=False,
            result=None,
            failure_reason=trace.terminal_error or "Unknown failure",
            resilience_metadata=meta,
        )


class LogAndFailRecoveryStrategy(BaseRecoveryStrategy):
    """Recovery strategy that logs the trace and returns failed recovery result."""

    def validate_compatibility(self, definition: RecoveryDefinition) -> None:
        """Validates configuration compatibility."""
        if not isinstance(definition, RecoveryDefinition):
            raise PipelineResilienceConfigurationError("Invalid definition type.")

    def recover(
        self,
        request: PipelineExecutionRequest,
        trace: RetryExecutionTrace,
        definition: RecoveryDefinition,
        pipeline_profile_id: str,
        resilience_profile_id: str,
        timeout_enforced: bool,
    ) -> PipelineRecoveryResult:
        """Logs the failure details and returns failed recovery result."""
        logger.error(
            "Resilience recovery invoked for execution %s "
            "after %s failed attempts. Trace: %s",
            trace.execution_id,
            trace.total_attempts,
            trace.model_dump_json(),
        )
        meta = ResilienceRuntimeMetadata(
            pipeline_profile_id=pipeline_profile_id,
            resilience_profile_id=resilience_profile_id,
            timeout_enforced=timeout_enforced,
            retry_trace=trace,
            recovery_invoked=True,
            recovery_strategy_id=definition.strategy_id,
            observed_at=datetime.now(timezone.utc),
        )
        return PipelineRecoveryResult(
            recovery_strategy_id=definition.strategy_id,
            succeeded=False,
            result=None,
            failure_reason=trace.terminal_error or "Exhausted retries",
            resilience_metadata=meta,
        )
