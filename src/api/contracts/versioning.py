"""Versioning and profile models for the API Contract layer."""

from enum import Enum
from typing import Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from src.core.exceptions import (
    ApiContractProfileNotFoundError,
    DuplicateApiContractProfileError,
)


class ApiVersionId(str, Enum):
    """Strongly typed API version enumeration."""

    V1 = "v1"
    V2 = "v2"
    V1_ALPHA = "v1-alpha"
    V1_BETA = "v1-beta"


class ApiContractDefinition(BaseModel):
    """Immutable definition of an API contract's capabilities."""

    supported_versions: tuple[ApiVersionId, ...] = Field(default=(ApiVersionId.V1,))
    require_correlation_id: bool = Field(default=True)
    strict_validation: bool = Field(default=True)

    model_config = ConfigDict(frozen=True, extra="forbid")


class ApiContractProfile(BaseModel):
    """Immutable profile defining active API contract policies."""

    profile_id: str = Field(...)
    definition: ApiContractDefinition = Field(...)

    model_config = ConfigDict(frozen=True, extra="forbid")


class ApiContractRegistry:
    """Registry for managing API contract profiles."""

    def __init__(self, profiles: Sequence[ApiContractProfile]) -> None:
        """Initializes the registry with a sequence of profiles."""
        self._profiles: Mapping[str, ApiContractProfile] = {}
        for profile in profiles:
            if profile.profile_id in self._profiles:
                raise DuplicateApiContractProfileError(
                    f"Duplicate API contract profile ID: {profile.profile_id}"
                )
            # Safe because dict assignment is synchronous and isolated to init
            self._profiles[profile.profile_id] = profile

    def resolve(self, profile_id: str) -> ApiContractProfile:
        """Resolves an API contract profile by ID."""
        profile = self._profiles.get(profile_id)
        if not profile:
            raise ApiContractProfileNotFoundError(
                f"API contract profile not found: {profile_id}"
            )
        return profile
