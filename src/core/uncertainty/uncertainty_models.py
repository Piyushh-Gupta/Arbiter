"""Immutable domain models for the Uncertainty Estimation subsystem."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from src.core.failure_analysis.failure_analysis_models import FailureAnalysisResult


class UncertaintyLevel(str, Enum):
    """Closed vocabulary of framework-level uncertainty severity."""

    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


class UncertaintyMetadata(BaseModel):
    """Minimal immutable execution provenance attached to each UncertaintyResult."""

    strategy_id: str = Field(
        ...,
        description="Identifies which uncertainty estimator produced this result.",
    )

    model_config = ConfigDict(frozen=True)


class UncertaintyFactor(BaseModel):
    """Immutable representation of a specific identified driver of uncertainty."""

    code: str = Field(
        ...,
        description="Machine-readable identifier for the uncertainty driver (e.g., 'EPISTEMIC_DISAGREEMENT').",
    )
    description: str = Field(
        ...,
        description="Human-readable explanation of the uncertainty context.",
    )

    model_config = ConfigDict(frozen=True)


class UncertaintyDefinition(BaseModel):
    """Base immutable configuration for an uncertainty estimation strategy."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class UncertaintyResult(BaseModel):
    """Immutable, self-contained output of a single uncertainty estimation invocation."""

    level: UncertaintyLevel = Field(
        ...,
        description="The discrete uncertainty level assigned by the concrete estimator.",
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="The continuous normalized uncertainty score (0.0 = completely certain, 1.0 = completely uncertain).",
    )
    factors: frozenset[UncertaintyFactor] = Field(
        default_factory=frozenset,
        description="Optional immutable set of identifying models describing specific drivers of uncertainty.",
    )
    failure_analysis_result: FailureAnalysisResult = Field(
        ...,
        description="The originating failure analysis state, which natively embeds the verification result.",
    )
    metadata: UncertaintyMetadata = Field(
        ...,
        description="Minimal execution provenance for downstream observability.",
    )

    model_config = ConfigDict(frozen=True)
