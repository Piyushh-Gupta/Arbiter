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
