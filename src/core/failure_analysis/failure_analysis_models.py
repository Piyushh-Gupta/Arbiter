"""Immutable domain models for the Failure Analysis subsystem."""

from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from src.core.exceptions import (
    DuplicateFailureAnalysisProfileError,
    FailureAnalysisProfileNotFoundError,
)

if TYPE_CHECKING:
    from src.core.failure_analysis.base import BaseFailureAnalyzer
else:
    BaseFailureAnalyzer = Any

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


class FailureAnalysisProfile(BaseModel):
    """
    Immutable pairing of a failure analysis definition and its compatible analyzer.
    """

    profile_id: str = Field(
        ...,
        description="Unique identifier for this failure analysis profile.",
    )
    definition: FailureAnalysisDefinition = Field(
        ...,
        description="The strictly immutable configuration for this failure analysis strategy.",
    )
    analyzer: BaseFailureAnalyzer = Field(
        ...,
        description="The stateless executable strategy resolving the definition.",
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _validate_compatibility(self) -> "FailureAnalysisProfile":
        """Front-loads compatibility validation at profile construction."""
        self.analyzer.validate_compatibility(self.definition)
        return self


class FailureAnalysisProfileRegistry(BaseModel):
    """
    Immutable registry for managing failure analysis profiles.
    """

    profiles: tuple[FailureAnalysisProfile, ...] = Field(
        ...,
        min_length=1,
        description="The abstract collection of registered failure analysis profiles.",
    )

    _profile_index: dict[str, FailureAnalysisProfile] = PrivateAttr(
        default_factory=dict
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _build_and_validate_index(self) -> "FailureAnalysisProfileRegistry":
        index: dict[str, FailureAnalysisProfile] = {}
        for profile in self.profiles:
            if profile.profile_id in index:
                raise DuplicateFailureAnalysisProfileError(
                    f"Duplicate profile_id detected: {profile.profile_id}"
                )
            index[profile.profile_id] = profile

        # Bypass frozen constraint for private attribute initialization
        object.__setattr__(self, "_profile_index", index)
        return self

    def resolve(self, profile_id: str) -> FailureAnalysisProfile:
        """
        Resolves a profile statelessly in O(1) time.

        Args:
            profile_id: The requested profile identifier.

        Returns:
            FailureAnalysisProfile: The fully resolved and validated profile.

        Raises:
            FailureAnalysisProfileNotFoundError: If the profile_id is not registered.
        """
        if profile_id not in self._profile_index:
            raise FailureAnalysisProfileNotFoundError(
                f"Failure analysis profile not found: {profile_id}"
            )
        return self._profile_index[profile_id]
