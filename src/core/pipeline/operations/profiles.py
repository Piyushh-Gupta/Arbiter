from pydantic import BaseModel, ConfigDict, model_validator

from src.core.exceptions import (
    DuplicateOperationalProfileError,
    OperationalProfileNotFoundError,
    PipelineOperationalConfigurationError,
)
from src.core.pipeline.operations.operation_models import PipelineOperationalDefinition


class PipelineOperationalProfile(BaseModel):
    """Immutable profile defining operational configuration."""

    model_config = ConfigDict(frozen=True)

    profile_id: str
    definition: PipelineOperationalDefinition

    @model_validator(mode="after")
    def _validate_profile(self) -> "PipelineOperationalProfile":
        if not self.profile_id:
            raise PipelineOperationalConfigurationError("profile_id cannot be empty")
        if self.definition.startup_timeout_seconds < 0:
            raise PipelineOperationalConfigurationError(
                "startup_timeout_seconds must be >= 0"
            )
        if self.definition.shutdown_timeout_seconds < 0:
            raise PipelineOperationalConfigurationError(
                "shutdown_timeout_seconds must be >= 0"
            )
        if self.definition.health_check_timeout_seconds < 0:
            raise PipelineOperationalConfigurationError(
                "health_check_timeout_seconds must be >= 0"
            )
        return self


class PipelineOperationalProfileRegistry:
    """O(1) registry for managing PipelineOperationalProfiles."""

    def __init__(self, profiles: tuple[PipelineOperationalProfile, ...] = ()) -> None:
        self._profile_index: dict[str, PipelineOperationalProfile] = {}
        for profile in profiles:
            self.register(profile)

    def register(self, profile: PipelineOperationalProfile) -> None:
        """Registers a new profile statelessly."""
        if profile.profile_id in self._profile_index:
            raise DuplicateOperationalProfileError(
                f"Duplicate operational profile registered: {profile.profile_id}"
            )
        self._profile_index[profile.profile_id] = profile

    def resolve(self, profile_id: str) -> PipelineOperationalProfile:
        """Resolves a profile statelessly in O(1) time."""
        if profile_id not in self._profile_index:
            raise OperationalProfileNotFoundError(
                f"Operational profile not found: {profile_id}"
            )
        return self._profile_index[profile_id]

    @property
    def profiles(self) -> tuple[PipelineOperationalProfile, ...]:
        """Returns all registered profiles as an immutable tuple."""
        return tuple(self._profile_index.values())
