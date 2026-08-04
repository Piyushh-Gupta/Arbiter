"""Immutable Pydantic models for Pipeline Explainability & Audit Reporting (M5.5)."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from src.core.pipeline.pipeline_models import PipelineExecutionResult
from src.core.pipeline.telemetry.telemetry_models import PipelineTelemetrySnapshot


class PipelineExplanationFormat(str, Enum):
    MARKDOWN = "markdown"
    JSON = "json"
    TEXT = "text"


class PipelineExplanationDefinition(BaseModel):
    """Immutable configuration for pipeline explanation generation."""

    template_format: PipelineExplanationFormat = Field(
        default=PipelineExplanationFormat.MARKDOWN
    )
    include_stage_breakdown: bool = Field(default=True)
    include_resilience_trace: bool = Field(default=True)
    include_telemetry_summary: bool = Field(default=True)
    include_configuration_fingerprint: bool = Field(default=True)
    metadata: dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(frozen=True)


class PipelineExplanationInput(BaseModel):
    """Immutable read-only evidence bundle consumed by explanation strategies."""

    execution_result: PipelineExecutionResult = Field(...)
    telemetry_snapshot: PipelineTelemetrySnapshot | None = Field(default=None)
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class PipelineExecutionSummary(BaseModel):
    """Immutable structured summary of a pipeline execution."""

    outcome: str = Field(..., min_length=1)  # "SUCCESS" or "FAILURE"
    total_latency_ms: float = Field(..., ge=0.0)
    stage_count: int = Field(..., ge=0)
    configuration_fingerprint: str = Field(..., min_length=1)
    model_config = ConfigDict(frozen=True)


class PipelineTelemetryContext(BaseModel):
    """Immutable telemetry summary context for the explanation."""

    total_executions: int | None = Field(default=None, ge=0)
    overall_success_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    model_config = ConfigDict(frozen=True)


class PipelineStageExplanation(BaseModel):
    """Immutable structured explanation for a single pipeline stage."""

    stage_id: str = Field(..., min_length=1)
    profile_id: str = Field(..., min_length=1)
    latency_ms: float = Field(..., ge=0.0)
    success: bool = Field(...)
    latency_percentile_rank: float | None = Field(default=None, ge=0.0, le=100.0)
    observation: str = Field(default="")
    trace_id: str = Field(..., min_length=1)
    model_config = ConfigDict(frozen=True)


class PipelineDecisionTrace(BaseModel):
    """Immutable structured trace of the pipeline's resilience decision path."""

    total_attempts: int = Field(..., ge=0)
    succeeded_on_attempt: int | None = Field(default=None, ge=1)
    timeout_enforced: bool = Field(...)
    recovery_invoked: bool = Field(...)
    recovery_strategy_id: str | None = Field(default=None)
    terminal_error: str | None = Field(default=None)
    total_retry_overhead_ms: float = Field(default=0.0, ge=0.0)
    trace_id: str = Field(..., min_length=1)
    model_config = ConfigDict(frozen=True)


class PipelineExecutionExplanation(BaseModel):
    """Immutable structured explanation for a complete pipeline execution."""

    execution_id: str = Field(..., min_length=1)
    pipeline_id: str = Field(..., min_length=1)
    claim_length: int = Field(..., ge=0)
    success: bool = Field(...)
    total_latency_ms: float = Field(..., ge=0.0)
    configuration_fingerprint: str = Field(...)
    schema_version: str = Field(...)
    execution_environment: str = Field(...)
    summary: PipelineExecutionSummary = Field(...)
    stage_explanations: tuple[PipelineStageExplanation, ...] = Field(
        default_factory=tuple
    )
    decision_trace: PipelineDecisionTrace | None = Field(default=None)
    telemetry_context: PipelineTelemetryContext = Field(...)
    metadata: dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(frozen=True)


class PipelineExplanationResult(BaseModel):
    """Immutable container for structured explanation and its rendered form."""

    explanation: PipelineExecutionExplanation = Field(...)
    rendered_format: str = Field(..., min_length=1)
    renderer_id: str = Field(..., min_length=1)
    strategy_id: str = Field(..., min_length=1)
    timestamp: str = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(frozen=True)


class PipelineAuditReport(BaseModel):
    """Immutable final audit report combining structured explanation with provenance."""

    execution_id: str = Field(..., min_length=1)
    explanation_result: PipelineExplanationResult = Field(...)
    profile_id: str = Field(..., min_length=1)
    schema_version: str = Field(default="1.0.0")
    generated_at: str = Field(..., min_length=1)
    model_config = ConfigDict(frozen=True)


class PipelineExplanationProfile(BaseModel):
    """Immutable association of profile_id with explanation definition and strategy."""

    profile_id: str = Field(..., min_length=1)
    definition: PipelineExplanationDefinition = Field(...)
    strategy: Any = Field(...)
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class PipelineExplanationProfileRegistry(BaseModel):
    """O(1) registry for pipeline explanation profile resolution."""

    profiles: tuple[PipelineExplanationProfile, ...] = Field(..., min_length=1)
    _profile_index: dict[str, PipelineExplanationProfile] = PrivateAttr(
        default_factory=dict
    )
    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def _build_and_validate_index(self) -> "PipelineExplanationProfileRegistry":
        from src.core.exceptions import DuplicatePipelineExplanationProfileError

        index: dict[str, PipelineExplanationProfile] = {}
        for p in self.profiles:
            profile_id = p.profile_id
            if profile_id in index:
                raise DuplicatePipelineExplanationProfileError(
                    f"Duplicate explanation profile ID detected: {profile_id}"
                )
            index[profile_id] = p
        object.__setattr__(self, "_profile_index", index)
        return self

    def resolve(self, profile_id: str) -> PipelineExplanationProfile:
        from src.core.exceptions import PipelineExplanationProfileNotFoundError

        if profile_id not in self._profile_index:
            raise PipelineExplanationProfileNotFoundError(
                f"Pipeline explanation profile not found: {profile_id}"
            )
        return self._profile_index[profile_id]

    def validate_compatibility(self, definition: PipelineExplanationDefinition) -> None:
        for p in self.profiles:
            p.strategy.validate_compatibility(definition)
