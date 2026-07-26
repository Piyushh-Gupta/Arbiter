"""Immutable domain models for the Verification subsystem."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from src.core.retrieval.retrieval_models import EvidenceBundle


class VerificationLabel(str, Enum):
    """Closed vocabulary of verification outcomes."""

    SUPPORTS = "SUPPORTS"
    REFUTES = "REFUTES"
    NOT_ENOUGH_INFO = "NOT_ENOUGH_INFO"


class VerificationMetadata(BaseModel):
    """Minimal immutable execution provenance attached to each VerificationResult."""

    strategy_id: str = Field(
        ...,
        description="Identifies which verifier produced this result (e.g., 'nli_fever').",
    )

    model_config = ConfigDict(frozen=True)


class VerificationResult(BaseModel):
    """Immutable, self-contained output of a single verification invocation."""

    label: VerificationLabel = Field(
        ...,
        description="The discrete verdict assigned to the claim.",
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional raw probability confidence score in [0.0, 1.0].",
    )
    evidence_bundle: EvidenceBundle = Field(
        ...,
        description="The immutable, originating bundle of evidence passages.",
    )
    metadata: VerificationMetadata = Field(
        ...,
        description="Minimal execution provenance for downstream observability.",
    )

    model_config = ConfigDict(frozen=True)


class VerificationDefinition(BaseModel):
    """Base immutable configuration for a verification strategy."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
