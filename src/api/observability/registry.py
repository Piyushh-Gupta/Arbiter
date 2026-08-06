"""MonitoringProfileRegistry for API Observability."""

import logging
from typing import Dict

from src.api.observability.telemetry_models import MonitoringProfile
from src.core.exceptions import (
    DuplicateMonitoringProfileError,
    MonitoringProfileNotFoundError,
    ObservabilityConfigurationError,
)

logger = logging.getLogger("arbiter.api.observability")


class MonitoringProfileRegistry:
    """O(1) thread-safe registry for MonitoringProfile instances."""

    def __init__(self) -> None:
        self._profiles: Dict[str, MonitoringProfile] = {}
        self._frozen: bool = False

    def register(self, profile: MonitoringProfile) -> None:
        """Registers a MonitoringProfile in the registry.

        Raises:
            ObservabilityConfigurationError: If the registry is frozen or profile validation fails.
            DuplicateMonitoringProfileError: If a profile with the same profile_id is already registered.
        """
        if self._frozen:
            raise ObservabilityConfigurationError(
                "Cannot register profiles after registry is frozen."
            )

        if profile.snapshot_interval_seconds <= 0:
            raise ObservabilityConfigurationError(
                f"Invalid snapshot_interval_seconds: {profile.snapshot_interval_seconds}. Must be > 0."
            )

        if profile.profile_id in self._profiles:
            raise DuplicateMonitoringProfileError(
                f"MonitoringProfile '{profile.profile_id}' is already registered."
            )

        self._profiles[profile.profile_id] = profile
        logger.info(f"Registered MonitoringProfile: {profile.profile_id}")

    def resolve(self, profile_id: str) -> MonitoringProfile:
        """Resolves a MonitoringProfile by ID in O(1) time.

        Raises:
            MonitoringProfileNotFoundError: If profile_id is not registered.
        """
        if profile_id not in self._profiles:
            raise MonitoringProfileNotFoundError(
                f"MonitoringProfile '{profile_id}' not found in registry."
            )
        return self._profiles[profile_id]

    def freeze(self) -> None:
        """Freezes the registry to prevent further profile additions."""
        self._frozen = True
        logger.info("MonitoringProfileRegistry frozen.")

    @property
    def is_frozen(self) -> bool:
        """Returns True if the registry is frozen."""
        return self._frozen
