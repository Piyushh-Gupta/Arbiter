"""Request models for the API Contract layer."""

from pydantic import BaseModel, ConfigDict, Field


class PaginationMetadata(BaseModel):
    """Immutable model representing pagination parameters."""

    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=50, ge=1, le=100)

    model_config = ConfigDict(frozen=True, extra="forbid")


class EvaluateClaimRequest(BaseModel):
    """Immutable model representing a request to evaluate a claim."""

    claim: str = Field(..., min_length=1)
    context: str | None = Field(default=None)
    metadata: dict[str, str] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True, extra="forbid")
