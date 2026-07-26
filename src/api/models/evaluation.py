"""Transport models for the evaluation API."""

from pydantic import BaseModel, ConfigDict, Field


class EvaluateClaimRequest(BaseModel):
    """External HTTP request model mapping directly to the PipelineExecutionRequest."""

    claim: str = Field(..., min_length=1)
    retrieval_profile_id: str = Field(...)
    verification_profile_id: str = Field(...)
    failure_analysis_profile_id: str = Field(...)
    uncertainty_profile_id: str = Field(...)
    decision_profile_id: str = Field(...)
    explanation_profile_id: str = Field(...)
    evaluation_profile_id: str = Field(...)

    model_config = ConfigDict(frozen=True)


class MetricDTO(BaseModel):
    """Transport model for an individual evaluation metric."""

    identifier: str
    title: str
    score: float


class EvaluationResponse(BaseModel):
    """External HTTP response model derived from the internal EvaluationResult."""

    metrics: list[MetricDTO]
