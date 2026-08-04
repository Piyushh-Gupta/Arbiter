"""Base protocols for Pipeline Explainability & Audit Reporting (M5.5)."""

from typing import Protocol, runtime_checkable

from src.core.pipeline.explainability.explainability_models import (
    PipelineExecutionExplanation,
    PipelineExplanationDefinition,
    PipelineExplanationInput,
)


@runtime_checkable
class BasePipelineExplanationStrategy(Protocol):
    """Stateless protocol for post-execution pipeline explanation strategies."""

    @property
    def strategy_id(self) -> str:
        """Unique identifier for the explanation strategy."""
        ...

    def validate_compatibility(self, definition: PipelineExplanationDefinition) -> None:
        """Validates compatibility with the explanation definition at startup."""
        ...

    def generate_explanation(
        self,
        input_data: PipelineExplanationInput,
        definition: PipelineExplanationDefinition,
    ) -> PipelineExecutionExplanation:
        """Generates a structured PipelineExecutionExplanation from immutable evidence."""
        ...


@runtime_checkable
class BasePipelineExplanationRenderer(Protocol):
    """Stateless protocol for rendering structured pipeline explanations."""

    @property
    def renderer_id(self) -> str:
        """Unique identifier for the renderer format."""
        ...

    def render(self, explanation: PipelineExecutionExplanation) -> str:
        """Renders a structured PipelineExecutionExplanation to its target format."""
        ...
