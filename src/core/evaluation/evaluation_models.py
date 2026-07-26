"""Immutable domain models for the Evaluation subsystem."""

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from src.core.exceptions import (
    DuplicateEvaluationProfileError,
    EvaluationProfileNotFoundError,
)
from src.core.explainability.explainability_models import ExplanationResult

if TYPE_CHECKING:
    from src.core.evaluation.base import BaseEvaluator
else:
    BaseEvaluator = Any


class EvaluationMetric(BaseModel):
    """An immutable, structured quantitative metric produced by an evaluator."""

    identifier: str = Field(
        ...,
        description="Machine-readable identifier for this metric (e.g., 'f1_score').",
    )
    title: str = Field(
        ...,
        description="Human-readable title for this metric.",
    )
    score: float = Field(
        ...,
        description="The continuous quantitative score for this metric.",
    )
    details: str | None = Field(
        default=None,
        description="Optional qualitative context explaining how the score was derived.",
    )

    model_config = ConfigDict(frozen=True)


class EvaluationDefinition(BaseModel):
    """Base immutable configuration for an evaluation strategy."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class RuleBasedEvaluationDefinition(EvaluationDefinition):
    """Configuration for the deterministic rule-based evaluator."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class EvaluationMetadata(BaseModel):
    """Minimal immutable execution provenance attached to each EvaluationResult."""

    strategy_id: str = Field(
        ...,
        description="Identifies which evaluation engine produced this result.",
    )

    model_config = ConfigDict(frozen=True)


class EvaluationResult(BaseModel):
    """Immutable, self-contained output of a single evaluation invocation."""

    metrics: tuple[EvaluationMetric, ...] = Field(
        ...,
        min_length=1,
        description="The generated quantitative metrics. Must contain at least one metric.",
    )
    explanation_result: ExplanationResult = Field(
        ...,
        description="The unbroken chain of prior pipeline state, capping the execution.",
    )
    metadata: EvaluationMetadata = Field(
        ...,
        description="Minimal execution provenance for downstream observability.",
    )

    model_config = ConfigDict(frozen=True)


class EvaluationProfile(BaseModel):
    """
    Immutable pairing of an evaluation policy configuration and its compatible strategy.
    """

    profile_id: str = Field(
        ...,
        description="Unique identifier for this evaluation profile.",
    )
    definition: EvaluationDefinition = Field(
        ...,
        description="The strictly immutable configuration for this evaluation strategy.",
    )
    engine: BaseEvaluator = Field(
        ...,
        description="The stateless executable strategy resolving the definition.",
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _validate_compatibility(self) -> "EvaluationProfile":
        """Front-loads compatibility validation at profile construction."""
        self.engine.validate_compatibility(self.definition)
        return self


class EvaluationProfileRegistry(BaseModel):
    """
    Immutable registry for managing evaluation profiles.
    """

    profiles: tuple[EvaluationProfile, ...] = Field(
        ...,
        min_length=1,
        description="The abstract collection of registered evaluation profiles.",
    )

    _profile_index: dict[str, EvaluationProfile] = PrivateAttr(default_factory=dict)

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _build_and_validate_index(self) -> "EvaluationProfileRegistry":
        """Builds an O(1) lookup index and statically detects duplicate profile IDs."""
        index: dict[str, EvaluationProfile] = {}
        for profile in self.profiles:
            if profile.profile_id in index:
                raise DuplicateEvaluationProfileError(
                    f"Duplicate profile_id detected: {profile.profile_id}"
                )
            index[profile.profile_id] = profile

        # Bypass frozen constraint to set the private index
        object.__setattr__(self, "_profile_index", index)
        return self

    def resolve(self, profile_id: str) -> EvaluationProfile:
        """Resolves a profile statelessly in O(1) time."""
        if profile_id not in self._profile_index:
            raise EvaluationProfileNotFoundError(
                f"Evaluation profile not found: {profile_id}"
            )
        return self._profile_index[profile_id]
