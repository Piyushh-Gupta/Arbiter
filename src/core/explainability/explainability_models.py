"""Immutable domain models for the Explainability subsystem."""

from pydantic import BaseModel, ConfigDict, Field

from src.core.decision.decision_models import DecisionResult


class ExplanationSection(BaseModel):
    """An immutable, structured segment of an explanation."""

    identifier: str = Field(
        ...,
        description="Machine-readable identifier for this section (e.g., 'evidence_trace').",
    )
    title: str = Field(
        ...,
        description="Human-readable title for this section.",
    )
    content: str = Field(
        ...,
        description="The textual content of the explanation section.",
    )

    model_config = ConfigDict(frozen=True)


class ExplanationDefinition(BaseModel):
    """Base immutable configuration for an explanation strategy."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class ExplanationMetadata(BaseModel):
    """Minimal immutable execution provenance attached to each ExplanationResult."""

    strategy_id: str = Field(
        ...,
        description="Identifies which explanation engine produced this result.",
    )

    model_config = ConfigDict(frozen=True)


class ExplanationResult(BaseModel):
    """Immutable, self-contained output of a single explanation invocation."""

    sections: tuple[ExplanationSection, ...] = Field(
        ...,
        min_length=1,
        description="The generated explanation segments. Must contain at least one section.",
    )
    decision_result: DecisionResult = Field(
        ...,
        description="The unbroken chain of prior pipeline state.",
    )
    metadata: ExplanationMetadata = Field(
        ...,
        description="Minimal execution provenance for downstream observability.",
    )

    model_config = ConfigDict(frozen=True)
