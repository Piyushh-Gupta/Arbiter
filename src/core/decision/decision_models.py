"""Immutable domain models for the Decision Engine subsystem."""

from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from src.core.exceptions import (
    DecisionProfileNotFoundError,
    DuplicateDecisionProfileError,
)
from src.core.uncertainty.uncertainty_models import UncertaintyResult

if TYPE_CHECKING:
    from src.core.decision.base import BaseDecisionEngine
else:
    BaseDecisionEngine = Any


class DecisionAction(str, Enum):
    """Closed vocabulary of final routing actions."""

    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    ESCALATE = "ESCALATE"
    ABSTAIN = "ABSTAIN"


class DecisionMetadata(BaseModel):
    """Minimal immutable execution provenance attached to each DecisionResult."""

    strategy_id: str = Field(
        ...,
        description="Identifies which decision engine produced this result.",
    )

    model_config = ConfigDict(frozen=True)


class DecisionDefinition(BaseModel):
    """Base immutable configuration for a decision policy."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class DecisionResult(BaseModel):
    """Immutable, self-contained output of a single decision invocation."""

    action: DecisionAction = Field(
        ...,
        description="The deterministic final routing action.",
    )
    rationale: str = Field(
        ...,
        description="A deterministic, human-readable justification for the selected action.",
    )
    uncertainty_result: UncertaintyResult = Field(
        ...,
        description="The unbroken chain of prior pipeline state.",
    )
    metadata: DecisionMetadata = Field(
        ...,
        description="Minimal execution provenance for downstream observability.",
    )

    model_config = ConfigDict(frozen=True)


class ThresholdDecisionDefinition(DecisionDefinition):
    """Configuration for threshold-based decision routing."""

    accept_max_uncertainty: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Maximum allowed uncertainty score to ACCEPT a supported claim.",
    )
    reject_max_uncertainty: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Maximum allowed uncertainty score to REJECT a refuted claim.",
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class DecisionProfile(BaseModel):
    """
    Immutable pairing of a decision policy configuration and its compatible strategy.
    """

    profile_id: str = Field(
        ...,
        description="Unique identifier for this decision profile.",
    )
    definition: DecisionDefinition = Field(
        ...,
        description="The strictly immutable configuration for this decision strategy.",
    )
    engine: BaseDecisionEngine = Field(
        ...,
        description="The stateless executable strategy resolving the definition.",
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _validate_compatibility(self) -> "DecisionProfile":
        """Front-loads compatibility validation at profile construction."""
        self.engine.validate_compatibility(self.definition)
        return self


class DecisionProfileRegistry(BaseModel):
    """
    Immutable registry for managing decision profiles.
    """

    profiles: tuple[DecisionProfile, ...] = Field(
        ...,
        min_length=1,
        description="The abstract collection of registered decision profiles.",
    )

    _profile_index: dict[str, DecisionProfile] = PrivateAttr(default_factory=dict)

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _build_and_validate_index(self) -> "DecisionProfileRegistry":
        """Builds an O(1) lookup index and statically detects duplicate profile IDs."""
        index: dict[str, DecisionProfile] = {}
        for profile in self.profiles:
            if profile.profile_id in index:
                raise DuplicateDecisionProfileError(
                    f"Duplicate profile_id detected: {profile.profile_id}"
                )
            index[profile.profile_id] = profile

        # Bypass frozen constraint to set the private index
        object.__setattr__(self, "_profile_index", index)
        return self

    def resolve(self, profile_id: str) -> DecisionProfile:
        """
        Resolves a profile statelessly in O(1) time.
        """
        if profile_id not in self._profile_index:
            raise DecisionProfileNotFoundError(
                f"Decision profile not found: {profile_id}"
            )
        return self._profile_index[profile_id]
