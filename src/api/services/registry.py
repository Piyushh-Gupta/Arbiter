"""Service layer registries."""

from src.api.services.base import BaseEvaluationService, BaseHealthService
from src.api.services.profiles import ServiceProfile
from src.core.exceptions import ArbiterError


class ServiceProfileNotFoundError(ArbiterError):
    """Raised when a service profile cannot be found."""


class DuplicateServiceProfileError(ArbiterError):
    """Raised when registering a duplicate service profile."""


class ServiceProfileRegistry:
    """Registry for service profiles."""

    def __init__(self, profiles: tuple[ServiceProfile, ...] = ()) -> None:
        self._profiles: dict[str, ServiceProfile] = {}
        for p in profiles:
            self.register(p)

    def register(self, profile: ServiceProfile) -> None:
        """Registers a service profile."""
        if profile.profile_id in self._profiles:
            raise DuplicateServiceProfileError(
                f"Duplicate profile: {profile.profile_id}"
            )
        self._profiles[profile.profile_id] = profile

    def resolve(self, profile_id: str) -> ServiceProfile:
        """Resolves a service profile by ID."""
        if profile_id not in self._profiles:
            raise ServiceProfileNotFoundError(f"Profile not found: {profile_id}")
        return self._profiles[profile_id]


class ServiceRegistry:
    """Registry holding instantiated services."""

    def __init__(
        self,
        evaluation_service: BaseEvaluationService,
        health_service: BaseHealthService,
    ) -> None:
        self._evaluation_service = evaluation_service
        self._health_service = health_service

    @property
    def evaluation_service(self) -> BaseEvaluationService:
        """Returns the evaluation service."""
        return self._evaluation_service

    @property
    def health_service(self) -> BaseHealthService:
        """Returns the health service."""
        return self._health_service
