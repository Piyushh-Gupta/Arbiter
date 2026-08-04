"""Dedicated pipeline explainability engine (M5.5)."""

from datetime import datetime, timezone

from src.core.exceptions import PipelineExplanationConfigurationError
from src.core.pipeline.explainability.base import BasePipelineExplanationRenderer
from src.core.pipeline.explainability.explainability_models import (
    PipelineAuditReport,
    PipelineExplanationInput,
    PipelineExplanationProfileRegistry,
    PipelineExplanationResult,
)


class PipelineExplanationEngine:
    """Stateless engine to coordinate post-execution pipeline explanation and auditing."""

    def __init__(
        self,
        registry: PipelineExplanationProfileRegistry,
        renderers: tuple[BasePipelineExplanationRenderer, ...] | None = None,
    ) -> None:
        self._registry = registry
        if renderers is None:
            from src.core.pipeline.explainability.rendering import (
                JsonPipelineRenderer,
                MarkdownPipelineRenderer,
                TextPipelineRenderer,
            )

            renderers = (
                MarkdownPipelineRenderer(),
                JsonPipelineRenderer(),
                TextPipelineRenderer(),
            )
        self._renderers = {r.renderer_id: r for r in renderers}

    def explain(
        self,
        input_data: PipelineExplanationInput,
        profile_id: str,
    ) -> PipelineExplanationResult:
        """Generates the structured pipeline explanation and renders it."""
        profile = self._registry.resolve(profile_id)

        explanation = profile.strategy.generate_explanation(
            input_data, profile.definition
        )

        renderer_id = profile.definition.template_format.value
        renderer = self._renderers.get(renderer_id)
        if not renderer:
            raise PipelineExplanationConfigurationError(
                f"Renderer not found for format: {renderer_id}"
            )

        rendered_content = renderer.render(explanation)

        timestamp = datetime.now(timezone.utc).isoformat()
        result = PipelineExplanationResult(
            explanation=explanation,
            rendered_format=renderer_id,
            renderer_id=renderer_id,
            strategy_id=profile.strategy.strategy_id,
            timestamp=timestamp,
            metadata={"rendered_content": rendered_content},
        )

        # Build the immutable PipelineAuditReport as required
        _ = PipelineAuditReport(
            execution_id=explanation.execution_id,
            explanation_result=result,
            profile_id=profile_id,
            schema_version="1.0.0",
            generated_at=timestamp,
        )

        return result

    def explain_with_audit(
        self,
        input_data: PipelineExplanationInput,
        profile_id: str,
    ) -> tuple[PipelineExplanationResult, PipelineAuditReport]:
        """Generates explanation result and the final audit report."""
        result = self.explain(input_data, profile_id)
        report = PipelineAuditReport(
            execution_id=result.explanation.execution_id,
            explanation_result=result,
            profile_id=profile_id,
            schema_version="1.0.0",
            generated_at=result.timestamp,
        )
        return result, report
