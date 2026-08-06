"""Unit tests for MonitoringProfileRegistry."""

import pytest

from src.api.observability.registry import MonitoringProfileRegistry
from src.api.observability.telemetry_models import MonitoringProfile
from src.core.exceptions import (
    DuplicateMonitoringProfileError,
    MonitoringProfileNotFoundError,
    ObservabilityConfigurationError,
)


def test_registry_register_and_resolve() -> None:
    registry = MonitoringProfileRegistry()
    profile = MonitoringProfile(profile_id="default", snapshot_interval_seconds=60.0)

    registry.register(profile)
    resolved = registry.resolve("default")
    assert resolved.profile_id == "default"


def test_registry_duplicate_registration() -> None:
    registry = MonitoringProfileRegistry()
    profile = MonitoringProfile(profile_id="default")
    registry.register(profile)

    with pytest.raises(DuplicateMonitoringProfileError):
        registry.register(profile)


def test_registry_not_found() -> None:
    registry = MonitoringProfileRegistry()
    with pytest.raises(MonitoringProfileNotFoundError):
        registry.resolve("non_existent")


def test_registry_freeze() -> None:
    registry = MonitoringProfileRegistry()
    profile1 = MonitoringProfile(profile_id="p1")
    registry.register(profile1)

    registry.freeze()
    assert registry.is_frozen is True

    profile2 = MonitoringProfile(profile_id="p2")
    with pytest.raises(ObservabilityConfigurationError):
        registry.register(profile2)


def test_registry_invalid_interval() -> None:
    registry = MonitoringProfileRegistry()
    with pytest.raises(ObservabilityConfigurationError):
        profile = MonitoringProfile.model_construct(
            profile_id="invalid", snapshot_interval_seconds=-1.0
        )
        registry.register(profile)
