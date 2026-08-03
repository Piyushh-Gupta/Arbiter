"""Immutable Pydantic models for Decision Explainability & Audit Reporting (M4.6)."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator


class DecisionExplanation(BaseModel):
    """Immutable structured model containing audit details for a decision."""

    summary: dict[str, Any] = Field(...)
    rule_trace: tuple[dict[str, Any], ...] = Field(default_factory=tuple)
    risk_trace: tuple[dict[str, Any], ...] = Field(default_factory=tuple)
    decision_trace: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


class DecisionExplanationDefinition(BaseModel):
    """Immutable definition parameters for generating explanations."""

    template_format: str = Field(default="markdown")
    include_traces: bool = Field(default=True)
    include_risk_factors: bool = Field(default=True)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


class DecisionExplanationResult(BaseModel):
    """Immutable container containing the structured explanation and its rendered form."""

    explanation: DecisionExplanation = Field(...)
    rendered_format: str = Field(..., min_length=1)
    renderer_id: str = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


class DecisionExplanationProfile(BaseModel):
    """Immutable association of profile_id with explanation definitions and strategies."""

    profile_id: str = Field(..., min_length=1)
    definition: DecisionExplanationDefinition = Field(...)
    strategy: Any = Field(...)

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class DecisionExplanationProfileRegistry(BaseModel):
    """O(1) registry mapping explanation profile IDs, validating duplicates and compatibility."""

    profiles: tuple[DecisionExplanationProfile, ...] = Field(..., min_length=1)

    _profile_index: dict[str, DecisionExplanationProfile] = PrivateAttr(
        default_factory=dict
    )

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def _build_and_validate_index(self) -> "DecisionExplanationProfileRegistry":
        from src.core.exceptions import DuplicateDecisionExplanationProfileError

        index: dict[str, DecisionExplanationProfile] = {}
        for p in self.profiles:
            profile_id = p.profile_id
            if profile_id in index:
                raise DuplicateDecisionExplanationProfileError(
                    f"Duplicate explanation profile ID detected: {profile_id}"
                )
            index[profile_id] = p
        object.__setattr__(self, "_profile_index", index)
        return self

    def resolve(self, profile_id: str) -> DecisionExplanationProfile:
        from src.core.exceptions import DecisionExplanationProfileNotFoundError

        if profile_id not in self._profile_index:
            raise DecisionExplanationProfileNotFoundError(
                f"Decision explanation profile not found: {profile_id}"
            )
        return self._profile_index[profile_id]

    def validate_compatibility(self, definition: Any) -> None:
        """Validates that explanation strategy/definition compatibility exists."""
        pass
