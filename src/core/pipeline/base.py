"""Base protocols for the Pipeline subsystem."""

from typing import Any, Protocol, runtime_checkable

from src.core.pipeline.pipeline_models import (
    PipelineDefinition,
    PipelineExecutionRequest,
    PipelineExecutionResult,
    PipelineStageDefinition,
)


@runtime_checkable
class BasePipelineOrchestrator(Protocol):
    """Stateless protocol for pipeline orchestrators."""

    def validate_compatibility(self, definition: PipelineDefinition) -> None:
        """Validates that the orchestrator is compatible with the given definition."""
        ...

    def execute(self, request: PipelineExecutionRequest) -> PipelineExecutionResult:
        """Executes the pipeline."""
        ...


@runtime_checkable
class BasePipelineStage(Protocol):
    """Stateless protocol for pipeline stages."""

    def validate_compatibility(self, definition: PipelineStageDefinition) -> None:
        """Validates that the stage is compatible with the given definition."""
        ...

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Executes the pipeline stage."""
        ...
