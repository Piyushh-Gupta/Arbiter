"""Immutable domain models for the Pipeline Orchestrator."""

from pydantic import BaseModel, ConfigDict, Field


class PipelineExecutionRequest(BaseModel):
    """Immutable request defining the input claim and the exact execution profile routing."""

    claim: str = Field(..., min_length=1, description="The textual claim to evaluate.")

    retrieval_profile_id: str = Field(...)
    verification_profile_id: str = Field(...)
    failure_analysis_profile_id: str = Field(...)
    uncertainty_profile_id: str = Field(...)
    decision_profile_id: str = Field(...)
    explanation_profile_id: str = Field(...)
    evaluation_profile_id: str = Field(...)

    model_config = ConfigDict(frozen=True)
