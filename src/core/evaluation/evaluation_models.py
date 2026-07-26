"""Immutable domain models for the Evaluation subsystem."""

from pydantic import BaseModel, ConfigDict, Field

from src.core.explainability.explainability_models import ExplanationResult


class EvaluationMetric(BaseModel):
    """An immutable, structured quantitative metric produced by an evaluator."""

    identifier: str = Field(
        ...,
        description="Machine-readable identifier for this metric (e.g., 'f1_score').",
    )
    title: str = Field(
        ...,
        description="Human-readable title for this metric.",
    )
    score: float = Field(
        ...,
        description="The continuous quantitative score for this metric.",
    )
    details: str | None = Field(
        default=None,
        description="Optional qualitative context explaining how the score was derived.",
    )

    model_config = ConfigDict(frozen=True)


class EvaluationDefinition(BaseModel):
    """Base immutable configuration for an evaluation strategy."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class EvaluationMetadata(BaseModel):
    """Minimal immutable execution provenance attached to each EvaluationResult."""

    strategy_id: str = Field(
        ...,
        description="Identifies which evaluation engine produced this result.",
    )

    model_config = ConfigDict(frozen=True)


class EvaluationResult(BaseModel):
    """Immutable, self-contained output of a single evaluation invocation."""

    metrics: tuple[EvaluationMetric, ...] = Field(
        ...,
        min_length=1,
        description="The generated quantitative metrics. Must contain at least one metric.",
    )
    explanation_result: ExplanationResult = Field(
        ...,
        description="The unbroken chain of prior pipeline state, capping the execution.",
    )
    metadata: EvaluationMetadata = Field(
        ...,
        description="Minimal execution provenance for downstream observability.",
    )

    model_config = ConfigDict(frozen=True)
