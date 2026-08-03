"""Immutable domain models for the Pipeline Orchestrator."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.core.evaluation.evaluation_models import EvaluationResult


class PipelineStageDefinition(BaseModel):
    """Configuration binding for a single pipeline stage."""

    stage_id: str = Field(..., min_length=1)
    profile_id: str = Field(..., min_length=1)
    enabled: bool = Field(default=True)
    model_config = ConfigDict(frozen=True)


class PipelineDefinition(BaseModel):
    """Immutable top-level pipeline configuration declaring the ordered stage profile binding."""

    pipeline_id: str = Field(..., min_length=1)
    retrieval_stage: PipelineStageDefinition = Field(...)
    verification_stage: PipelineStageDefinition = Field(...)
    failure_analysis_stage: PipelineStageDefinition = Field(...)
    uncertainty_stage: PipelineStageDefinition = Field(...)
    decision_stage: PipelineStageDefinition = Field(...)
    explanation_stage: PipelineStageDefinition = Field(...)
    evaluation_stage: PipelineStageDefinition = Field(...)
    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def _validate_unique_stages(self) -> "PipelineDefinition":
        stage_ids = [
            self.retrieval_stage.stage_id,
            self.verification_stage.stage_id,
            self.failure_analysis_stage.stage_id,
            self.uncertainty_stage.stage_id,
            self.decision_stage.stage_id,
            self.explanation_stage.stage_id,
            self.evaluation_stage.stage_id,
        ]
        if len(set(stage_ids)) != len(stage_ids):
            raise ValueError("Duplicate stage_id values found in PipelineDefinition.")
        return self


class PipelineStageMetadata(BaseModel):
    """Immutable per-stage execution metadata."""

    stage_id: str = Field(..., min_length=1)
    profile_id: str = Field(..., min_length=1)
    latency_ms: float = Field(..., ge=0.0)
    success: bool = Field(...)
    model_config = ConfigDict(frozen=True)


class PipelineRuntimeMetadata(BaseModel):
    """Immutable runtime metadata for the pipeline execution."""

    pipeline_version: str = Field(...)
    configuration_fingerprint: str = Field(...)
    schema_version: str = Field(...)
    execution_environment: str = Field(...)
    execution_timestamp: datetime = Field(...)
    model_config = ConfigDict(frozen=True)


class PipelineExecutionContext(BaseModel):
    """Immutable record of a complete pipeline execution, spanning all stages."""

    execution_id: str = Field(..., min_length=1)
    pipeline_id: str = Field(..., min_length=1)
    claim: str = Field(..., min_length=1)
    runtime_metadata: PipelineRuntimeMetadata = Field(...)
    stage_metadata: tuple[PipelineStageMetadata, ...] = Field(default_factory=tuple)
    total_latency_ms: float = Field(..., ge=0.0)
    success: bool = Field(...)
    model_config = ConfigDict(frozen=True)


class PipelineExecutionRequest(BaseModel):
    """Immutable execution request binding a claim to a PipelineProfile."""

    claim: str = Field(..., min_length=1)
    pipeline_profile_id: str = Field(..., min_length=1)
    model_config = ConfigDict(frozen=True)


class PipelineExecutionResult(BaseModel):
    """Immutable top-level result composing the domain EvaluationResult with execution context."""

    evaluation_result: EvaluationResult = Field(...)
    execution_context: PipelineExecutionContext = Field(...)
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
