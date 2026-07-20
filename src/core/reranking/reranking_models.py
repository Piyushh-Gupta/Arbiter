"""Immutable domain models for the Reranking subsystem."""

from pydantic import BaseModel, ConfigDict, Field


class RerankingDefinition(BaseModel):
    """Immutable configuration for a single reranking invocation."""

    top_k: int = Field(
        ...,
        gt=0,
        description="Maximum number of reranked passages to return.",
    )
    score_threshold: float | None = Field(
        default=None,
        description="Optional minimum cross-encoder score filter.",
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
