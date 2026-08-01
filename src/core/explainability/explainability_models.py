"""Immutable domain models for the Explainability subsystem."""

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from src.core.exceptions import (
    DuplicateExplanationProfileError,
    ExplanationProfileNotFoundError,
)

if TYPE_CHECKING:
    from src.core.explainability.base import BaseExplainer
else:
    BaseExplainer = Any


class ExplanationSection(BaseModel):
    """An immutable, structured segment of an explanation."""

    identifier: str = Field(
        ...,
        description="Machine-readable identifier for this section (e.g., 'evidence_trace').",
    )
    title: str = Field(
        ...,
        description="Human-readable title for this section.",
    )
    content: str = Field(
        ...,
        description="The textual content of the explanation section.",
    )

    model_config = ConfigDict(frozen=True)


class ExplanationDefinition(BaseModel):
    """Base immutable configuration for an explanation strategy."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class ExplanationMetadata(BaseModel):
    """Minimal immutable execution provenance attached to each ExplanationResult."""

    strategy_id: str = Field(
        ...,
        description="Identifies which explanation engine produced this result.",
    )

    model_config = ConfigDict(frozen=True)


class ExplanationResult(BaseModel):
    """Immutable, self-contained output of a single explanation invocation."""

    sections: tuple[ExplanationSection, ...] = Field(
        ...,
        min_length=1,
        description="The generated explanation segments. Must contain at least one section.",
    )
    decision_result: Any = Field(
        default=None,
        description="The unbroken chain of prior pipeline state.",
    )
    metadata: ExplanationMetadata = Field(
        ...,
        description="Minimal execution provenance for downstream observability.",
    )
    verification_result: Any | None = Field(
        default=None,
        description="Optional verification result associated with this explanation.",
    )
    calibration_result: Any | None = Field(
        default=None,
        description="Optional calibration result associated with this explanation.",
    )
    evidence_attribution: Any | None = Field(
        default=None,
        description="Optional evidence attribution associated with this explanation.",
    )
    decision_trace: Any | None = Field(
        default=None,
        description="Optional decision trace associated with this explanation.",
    )
    contribution_analysis: Any | None = Field(
        default=None,
        description="Optional contribution analysis associated with this explanation.",
    )
    explanation_trace: Any | None = Field(
        default=None,
        description="Optional explanation trace associated with this explanation.",
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class RuleBasedExplanationDefinition(ExplanationDefinition):
    """Configuration for the deterministic rule-based explainer."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class ExplanationProfile(BaseModel):
    """
    Immutable pairing of an explanation policy configuration and its compatible strategy.
    """

    profile_id: str = Field(
        ...,
        description="Unique identifier for this explanation profile.",
    )
    definition: ExplanationDefinition = Field(
        ...,
        description="The strictly immutable configuration for this explanation strategy.",
    )
    engine: Any = Field(
        ...,
        description="The stateless executable strategy resolving the definition.",
    )
    verification_profile_id: str | None = Field(
        default=None, description="Optional referenced verification profile ID."
    )
    calibration_profile_id: str | None = Field(
        default=None, description="Optional referenced calibration profile ID."
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _validate_compatibility(self) -> "ExplanationProfile":
        """Front-loads compatibility validation at profile construction."""
        if hasattr(self.engine, "validate_compatibility"):
            self.engine.validate_compatibility(self.definition)
        return self


class ExplanationProfileRegistry(BaseModel):
    """
    Immutable registry for managing explanation profiles.
    """

    profiles: tuple[ExplanationProfile, ...] = Field(
        ...,
        min_length=1,
        description="The abstract collection of registered explanation profiles.",
    )

    _profile_index: dict[str, ExplanationProfile] = PrivateAttr(default_factory=dict)

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _build_and_validate_index(self) -> "ExplanationProfileRegistry":
        """Builds an O(1) lookup index and statically detects duplicate profile IDs."""
        index: dict[str, ExplanationProfile] = {}
        for profile in self.profiles:
            if profile.profile_id in index:
                raise DuplicateExplanationProfileError(
                    f"Duplicate profile_id detected: {profile.profile_id}"
                )
            index[profile.profile_id] = profile

        # Bypass frozen constraint to set the private index
        object.__setattr__(self, "_profile_index", index)
        return self

    def resolve(self, profile_id: str) -> ExplanationProfile:
        """
        Resolves a profile statelessly in O(1) time.
        """
        if profile_id not in self._profile_index:
            raise ExplanationProfileNotFoundError(
                f"Explanation profile not found: {profile_id}"
            )
        return self._profile_index[profile_id]
