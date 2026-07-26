"""Immutable domain models for the Uncertainty Estimation subsystem."""

from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from src.core.exceptions import (
    DuplicateUncertaintyProfileError,
    UncertaintyProfileNotFoundError,
)
from src.core.failure_analysis.failure_analysis_models import (
    FailureAnalysisResult,
    FailureSeverity,
)

if TYPE_CHECKING:
    from src.core.uncertainty.base import BaseUncertaintyEstimator
else:
    BaseUncertaintyEstimator = Any


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


class ConfidenceUncertaintyDefinition(UncertaintyDefinition):
    """Configuration for confidence-based uncertainty estimation."""

    none_threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Maximum uncertainty score to be classified as NONE.",
    )
    low_threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Maximum uncertainty score to be classified as LOW.",
    )
    medium_threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Maximum uncertainty score to be classified as MEDIUM.",
    )
    high_threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Maximum uncertainty score to be classified as HIGH.",
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _validate_threshold_ordering(self) -> "ConfidenceUncertaintyDefinition":
        """Ensures thresholds are strictly increasing."""
        if not (
            self.none_threshold
            < self.low_threshold
            < self.medium_threshold
            < self.high_threshold
        ):
            raise ValueError(
                "Thresholds must be strictly increasing: none < low < medium < high."
            )
        return self


class FailureAwareUncertaintyDefinition(ConfidenceUncertaintyDefinition):
    """Configuration for failure-aware uncertainty estimation."""

    severity_penalties: dict[FailureSeverity, float] = Field(
        default_factory=dict,
        description="Maps overall FailureSeverity to a certainty penalty [0.0, 1.0].",
    )
    flag_penalties: dict[str, float] = Field(
        default_factory=dict,
        description="Maps specific FailureFlag.code identifiers (e.g., 'CONTRADICTORY_EVIDENCE') to a certainty penalty [0.0, 1.0].",
    )

    @model_validator(mode="after")
    def _validate_penalties(self) -> "FailureAwareUncertaintyDefinition":
        """Ensures all penalties are within [0.0, 1.0]."""
        for sev, pen in self.severity_penalties.items():
            if not (0.0 <= pen <= 1.0):
                raise ValueError(f"Severity penalty for {sev} must be in [0.0, 1.0].")
        for flag, pen in self.flag_penalties.items():
            if not (0.0 <= pen <= 1.0):
                raise ValueError(f"Flag penalty for {flag} must be in [0.0, 1.0].")
        return self


class UncertaintyProfile(BaseModel):
    """
    Immutable pairing of an uncertainty estimation definition and its compatible strategy.
    """

    profile_id: str = Field(
        ...,
        description="Unique identifier for this uncertainty profile.",
    )
    definition: UncertaintyDefinition = Field(
        ...,
        description="The strictly immutable configuration for this uncertainty strategy.",
    )
    estimator: BaseUncertaintyEstimator = Field(
        ...,
        description="The stateless executable strategy resolving the definition.",
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _validate_compatibility(self) -> "UncertaintyProfile":
        """Front-loads compatibility validation at profile construction."""
        self.estimator.validate_compatibility(self.definition)
        return self


class UncertaintyProfileRegistry(BaseModel):
    """
    Immutable registry for managing uncertainty profiles.
    """

    profiles: tuple[UncertaintyProfile, ...] = Field(
        ...,
        min_length=1,
        description="The abstract collection of registered uncertainty profiles.",
    )

    _profile_index: dict[str, UncertaintyProfile] = PrivateAttr(default_factory=dict)

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _build_and_validate_index(self) -> "UncertaintyProfileRegistry":
        """Builds an O(1) lookup index and statically detects duplicate profile IDs."""
        index: dict[str, UncertaintyProfile] = {}
        for profile in self.profiles:
            if profile.profile_id in index:
                raise DuplicateUncertaintyProfileError(
                    f"Duplicate profile_id detected: {profile.profile_id}"
                )
            index[profile.profile_id] = profile

        # Bypass frozen constraint to set the private index
        object.__setattr__(self, "_profile_index", index)
        return self

    def resolve(self, profile_id: str) -> UncertaintyProfile:
        """
        Resolves a profile statelessly in O(1) time.
        """
        if profile_id not in self._profile_index:
            raise UncertaintyProfileNotFoundError(
                f"Uncertainty profile not found: {profile_id}"
            )
        return self._profile_index[profile_id]
