"""Immutable domain models for the Failure Analysis subsystem."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from src.core.verification.verification_models import VerificationResult


class FailureSeverity(str, Enum):
    """Closed vocabulary of framework-level severity levels."""

    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class FailureFlag(BaseModel):
    """Immutable representation of a single identified failure mode."""

    code: str = Field(
        ...,
        description="Machine-readable identifier for the failure (e.g., 'EMPTY_BUNDLE').",
    )
    description: str = Field(
        ...,
        description="Human-readable explanation of the failure context.",
    )

    model_config = ConfigDict(frozen=True)


class FailureMetadata(BaseModel):
    """Minimal immutable execution provenance attached to each FailureAnalysisResult."""

    strategy_id: str = Field(
        ...,
        description="Identifies which failure analyzer produced this result.",
    )

    model_config = ConfigDict(frozen=True)


class FailureAnalysisDefinition(BaseModel):
    """Base immutable configuration for a failure analysis strategy."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class FailureAnalysisResult(BaseModel):
    """Immutable, self-contained output of a single failure analysis invocation."""

    failure_flags: frozenset[FailureFlag] = Field(
        ...,
        description="The immutable set of failure mode objects identified during analysis.",
    )
    severity: FailureSeverity = Field(
        ...,
        description="The aggregate severity level assigned by the concrete analyzer.",
    )
    verification_result: VerificationResult = Field(
        ...,
        description="The originating complete pipeline state.",
    )
    metadata: FailureMetadata = Field(
        ...,
        description="Minimal execution provenance for downstream observability.",
    )

    model_config = ConfigDict(frozen=True)


class RetrievalFailureAnalysisDefinition(FailureAnalysisDefinition):
    """Configuration for retrieval failure analysis."""

    min_passages: int = Field(
        ...,
        gt=0,
        description="Minimum number of passages required for reliable verification.",
    )
    min_score_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional minimum acceptable retrieval score.",
    )
    min_unique_documents: int | None = Field(
        default=None,
        gt=0,
        description="Optional minimum number of distinct source documents required.",
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class VerificationFailureAnalysisDefinition(FailureAnalysisDefinition):
    """Configuration for verification failure analysis."""

    min_confidence_threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Minimum acceptable confidence score for the verification verdict.",
    )
    flag_nei_verdict: bool = Field(
        default=True,
        description="Whether to flag a NOT_ENOUGH_INFO verdict as a pipeline failure.",
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class ContradictionAnalysisDefinition(FailureAnalysisDefinition):
    """Configuration for contradiction failure analysis."""

    min_passage_label_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Minimum confidence a passage label must achieve to be counted as supporting or refuting.",
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
