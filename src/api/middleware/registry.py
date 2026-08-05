"""Registry for middleware profiles."""

from typing import Sequence

from src.api.middleware.middleware_models import MiddlewareProfile
from src.core.exceptions import (
    DuplicateMiddlewareProfileError,
    MiddlewareProfileNotFoundError,
)


class MiddlewareProfileRegistry:
    """Registry for managing middleware configuration profiles."""

    def __init__(self, profiles: Sequence[MiddlewareProfile]) -> None:
        """Initializes the registry with a sequence of profiles."""
        self._profiles: dict[str, MiddlewareProfile] = {}
        for profile in profiles:
            if profile.profile_id in self._profiles:
                raise DuplicateMiddlewareProfileError(
                    f"Duplicate middleware profile ID: {profile.profile_id}"
                )
            self._profiles[profile.profile_id] = profile

    def resolve(self, profile_id: str) -> MiddlewareProfile:
        """Resolves a middleware profile by ID in O(1) time."""
        profile = self._profiles.get(profile_id)
        if not profile:
            raise MiddlewareProfileNotFoundError(
                f"Middleware profile not found: {profile_id}"
            )
        return profile
