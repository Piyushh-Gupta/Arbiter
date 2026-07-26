"""Immutable domain models for the Verification subsystem."""

from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

if TYPE_CHECKING:
    from src.core.verification.base import BaseVerifier
else:
    BaseVerifier = Any


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


class NLIVerificationDefinition(VerificationDefinition):
    """Immutable configuration for an NLI verification invocation."""

    top_k: int = Field(
        ...,
        gt=0,
        description="Maximum number of passages to evaluate from the bundle.",
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class VerificationProfile(BaseModel):
    """Immutable reusable wrapper binding a verification definition to its execution strategy."""

    profile_id: str = Field(
        ..., description="Unique identifier for this verification profile."
    )
    definition: VerificationDefinition = Field(
        ...,
        description="The strictly immutable configuration for this verification strategy.",
    )
    verifier: "BaseVerifier" = Field(
        ..., description="The stateless executable strategy resolving the definition."
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _validate_compatibility(self) -> "VerificationProfile":
        """Statically verifies compatibility between the definition and strategy upon construction."""
        self.verifier.validate_compatibility(self.definition)
        return self


class VerificationProfileRegistry(BaseModel):
    """Immutable namespace for securely resolving named verification profiles."""

    profiles: tuple[VerificationProfile, ...] = Field(
        ...,
        min_length=1,
        description="The abstract collection of registered verification profiles.",
    )

    _profile_index: dict[str, VerificationProfile] = PrivateAttr(default_factory=dict)

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _build_and_validate_index(self) -> "VerificationProfileRegistry":
        from src.core.exceptions import DuplicateVerificationProfileError

        index: dict[str, VerificationProfile] = {}
        for profile in self.profiles:
            if profile.profile_id in index:
                raise DuplicateVerificationProfileError(
                    f"Duplicate verification profile identifier: {profile.profile_id}"
                )
            index[profile.profile_id] = profile

        # Bypass Pydantic's frozen constraint to initialize the O(1) private lookup table
        object.__setattr__(self, "_profile_index", index)
        return self

    def resolve(self, profile_id: str) -> VerificationProfile:
        """Resolves a profile statelessly in O(1) time."""
        from src.core.exceptions import VerificationProfileNotFoundError

        if profile_id not in self._profile_index:
            raise VerificationProfileNotFoundError(
                f"Verification profile not found: {profile_id}"
            )
        return self._profile_index[profile_id]
