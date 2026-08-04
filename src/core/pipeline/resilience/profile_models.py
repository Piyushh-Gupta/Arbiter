"""Profile and registry models for pipeline resilience."""

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from src.core.exceptions import (
    DuplicateResilienceProfileError,
    PipelineResilienceConfigurationError,
    ResilienceProfileNotFoundError,
)
from src.core.pipeline.resilience.base import BaseRecoveryStrategy
from src.core.pipeline.resilience.resilience_models import PipelineResilienceDefinition


class PipelineResilienceProfile(BaseModel):
    """Immutable binding of a PipelineResilienceDefinition to a recovery strategy."""

    profile_id: str = Field(..., min_length=1)
    definition: PipelineResilienceDefinition = Field(...)
    recovery_strategy: BaseRecoveryStrategy = Field(...)
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _validate_compatibility(self) -> "PipelineResilienceProfile":
        """Validates strategy compatibility at construction."""
        try:
            self.recovery_strategy.validate_compatibility(self.definition.recovery)
        except PipelineResilienceConfigurationError:
            raise
        except Exception as e:
            raise PipelineResilienceConfigurationError(
                f"Resilience profile {self.profile_id} is incompatible: {e}"
            ) from e
        return self


class PipelineResilienceProfileRegistry(BaseModel):
    """Registry providing O(1) resilience profile resolution."""

    profiles: tuple[PipelineResilienceProfile, ...] = Field(..., min_length=1)
    _profile_index: dict[str, PipelineResilienceProfile] = PrivateAttr(
        default_factory=dict
    )
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _build_index(self) -> "PipelineResilienceProfileRegistry":
        """Builds the internal dict index and detects duplicate profile_ids."""
        index = {}
        for profile in self.profiles:
            if profile.profile_id in index:
                raise DuplicateResilienceProfileError(
                    f"Duplicate resilience profile_id: {profile.profile_id}"
                )
            index[profile.profile_id] = profile
        self._profile_index = index
        return self

    def resolve(self, profile_id: str) -> PipelineResilienceProfile:
        """Resolves a resilience profile by ID.

        Raises:
            ResilienceProfileNotFoundError: If profile_id is absent.
        """
        if profile_id not in self._profile_index:
            raise ResilienceProfileNotFoundError(
                f"Resilience profile '{profile_id}' not found in registry."
            )
        return self._profile_index[profile_id]
